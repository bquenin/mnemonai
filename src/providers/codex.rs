use crate::claude::{
    AssistantMessage, ContentBlock, LogEntry, UserContent, UserMessage, extract_text_from_blocks,
};
use crate::cli::DebugLevel;
use crate::conversation_index::{
    CachedFileConversation, SourceFingerprint, delete_conversation, fingerprint_from_metadata,
    load_provider_cache, prune_conversations, save_conversations,
};
use crate::debug;
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, ParseError, ProviderKind, format_short_name_from_path,
};
use chrono::{DateTime, FixedOffset, Local};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::SystemTime;

pub struct CodexProvider {
    codex_home: PathBuf,
}

/// What a parse pass is being run for.
///
/// The two consumers of a transcript need disjoint work: the startup scan wants
/// only the [`Conversation`] summary (preview/full-text/counts/timestamps),
/// while the viewer wants only the reconstructed [`LogEntry`] list. Building
/// both on every pass was the bulk of the wasted work — tool outputs (the
/// largest payloads in a transcript) were cloned into entries and immediately
/// dropped at startup, and the full conversation text was assembled and dropped
/// when reading entries. The mode lets each pass skip the other's output.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParseMode {
    /// Startup listing: produce a `Conversation`, skip all `LogEntry` work.
    Scan,
    /// Viewer: produce `LogEntry` values, skip conversation/preview text work.
    Entries,
}

/// The cheap head of every record. Deserializing this first lets serde skip the
/// rest of the line when scan mode only needs the record's type and timestamp:
/// serde_json walks a multi-MB `function_call_output` payload's fields but never
/// allocates them into a `Value` tree the way `from_str::<Value>` would.
///
/// The nested [`PayloadHead`] captures the inner discriminator for the wrapped
/// (`response_item` / `event_msg`) records where the meaningful type lives under
/// `payload.type`, again without materializing the large payload fields.
#[derive(Deserialize)]
struct RecordHead {
    // Lenient, so the head matches the original `Value::get(..).and_then(as_str)`
    // dispatch exactly: a field of the wrong JSON type is treated as absent
    // rather than failing the whole parse (which would spuriously flag the line
    // as a parse error the Value path never reported).
    #[serde(rename = "type", default, deserialize_with = "lenient_string")]
    kind: Option<String>,
    #[serde(default, deserialize_with = "lenient_string")]
    timestamp: Option<String>,
    #[serde(default, deserialize_with = "lenient_string")]
    id: Option<String>,
    // A plain nested struct: serde_json walks the payload object and pulls only
    // `type`, skipping (never allocating) large fields like `output`. A null
    // payload deserializes to None; the rare non-object payload fails the head
    // parse and is handled by the Value fallback, matching the old dispatch
    // (which found no `type` on a non-object payload and did nothing).
    #[serde(default)]
    payload: Option<PayloadHead>,
}

#[derive(Deserialize)]
struct PayloadHead {
    #[serde(rename = "type", default, deserialize_with = "lenient_string")]
    kind: Option<String>,
}

/// Deserialize a field as `Some(string)` only when it is a JSON string, and as
/// `None` for null or any other type — mirroring `Value::as_str`.
fn lenient_string<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match serde::Deserialize::deserialize(deserializer)? {
        Value::String(text) => Ok(Some(text)),
        _ => Ok(None),
    }
}

impl RecordHead {
    /// Legacy session-metadata records carry an `id` and `timestamp` but no
    /// `type` (and no `payload`); mirror [`is_legacy_session_metadata`] on the
    /// head alone so scan mode can detect them without a full parse.
    fn is_legacy_session_metadata(&self) -> bool {
        self.id.is_some()
            && self.timestamp.is_some()
            && self.kind.is_none()
            && self.payload.is_none()
    }

    /// The record's meaningful item type, mirroring the runtime dispatch:
    /// `session_meta` / `turn_context` are outer-typed; every other record's
    /// discriminator lives under `payload.type` when wrapped, else at the top
    /// level (legacy records).
    fn item_kind(&self) -> Option<&str> {
        match self.kind.as_deref() {
            Some(outer @ ("session_meta" | "turn_context")) => Some(outer),
            _ => self
                .payload
                .as_ref()
                .and_then(|payload| payload.kind.as_deref())
                .or(self.kind.as_deref()),
        }
    }
}

struct ParsedCodexTranscript {
    conversation: Option<Conversation>,
    entries: Vec<LogEntry>,
}

#[derive(Default)]
struct CodexParseState {
    id: Option<String>,
    cwd: Option<PathBuf>,
    workspace_roots: Vec<PathBuf>,
    model: Option<String>,
    total_tokens: u64,
    metadata_timestamp: Option<DateTime<FixedOffset>>,
    // The first timestamp is parsed eagerly but only until the first record
    // whose timestamp parses (usually the very first line), matching the old
    // "first successfully parsed" semantics; once set it is never re-derived.
    // The last timestamp is deferred: every record carrying a timestamp string
    // overwrites `last_timestamp_raw` in place (reusing its heap allocation),
    // and it is parsed exactly once, at EOF, in `finalize_timestamps` — instead
    // of RFC3339-parsing every record just to keep the final one.
    first_timestamp: Option<DateTime<FixedOffset>>,
    last_timestamp: Option<DateTime<FixedOffset>>,
    last_timestamp_raw: String,
    has_last_timestamp: bool,
    entries: Vec<LogEntry>,
    all_parts: Vec<String>,
    message_count: usize,
    parse_errors: Vec<ParseError>,
}

enum ConversationLoad {
    Cached(Conversation),
    Fresh(Conversation, SourceFingerprint),
}

impl ConversationLoad {
    fn into_conversation(self) -> Conversation {
        match self {
            Self::Cached(conversation) | Self::Fresh(conversation, _) => conversation,
        }
    }

    /// Borrow a freshly parsed conversation and its fingerprint for saving,
    /// letting the index be written from references without cloning.
    fn as_fresh(&self) -> Option<(&Conversation, SourceFingerprint)> {
        match self {
            Self::Fresh(conversation, fingerprint) => Some((conversation, *fingerprint)),
            Self::Cached(_) => None,
        }
    }
}

impl CodexProvider {
    pub fn new() -> Self {
        Self {
            codex_home: resolve_codex_home(std::env::var_os("CODEX_HOME"), home::home_dir()),
        }
    }

    fn sessions_root(&self) -> Option<PathBuf> {
        if self.codex_home.as_os_str().is_empty() {
            return None;
        }
        Some(self.codex_home.join("sessions"))
    }

    fn load_all_conversations(
        &self,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) -> Result<Vec<Conversation>> {
        let Some(root) = self.sessions_root() else {
            return Ok(Vec::new());
        };
        if !root.exists() {
            return Ok(Vec::new());
        }

        let files = collect_session_files(&root);
        let cache = Mutex::new(load_provider_cache(ProviderKind::Codex, show_last));
        let loaded: Vec<ConversationLoad> = files
            .into_par_iter()
            .filter_map(|path| load_one(path, show_last, &cache, debug_level))
            .collect();

        // Save the freshly parsed conversations from references, before we move
        // them into the returned list — no per-conversation clone.
        save_conversations(
            ProviderKind::Codex,
            show_last,
            loaded.iter().filter_map(ConversationLoad::as_fresh),
        );

        // Entries no session file claimed belong to files that no longer exist.
        let stale: Vec<PathBuf> = cache
            .into_inner()
            .map(|cache| cache.into_keys().collect())
            .unwrap_or_default();
        prune_conversations(ProviderKind::Codex, &stale);

        let mut conversations: Vec<Conversation> = loaded
            .into_iter()
            .map(ConversationLoad::into_conversation)
            .collect();
        conversations.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(conversations)
    }

    /// Stream conversations to the TUI in incremental batches so Codex history
    /// paints as it loads instead of only after the whole provider finishes.
    /// Sessions live under `sessions/YYYY/MM/DD/`, so each day directory is a
    /// natural batch boundary. Each batch's fresh conversations are persisted
    /// (from references) before the batch is sent — the same fingerprint the
    /// sync path would write — and pruning of vanished files happens once at
    /// the end, after every directory has consumed its cache entries.
    fn stream_all_conversations(
        &self,
        tx: &Sender<LoaderMessage>,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) {
        let Some(root) = self.sessions_root() else {
            let _ = tx.send(LoaderMessage::Done);
            return;
        };
        if !root.exists() {
            let _ = tx.send(LoaderMessage::Done);
            return;
        }

        let cache = Mutex::new(load_provider_cache(ProviderKind::Codex, show_last));

        for (_dir, files) in collect_session_batch_dirs(&root) {
            let loaded: Vec<ConversationLoad> = files
                .into_par_iter()
                .filter_map(|path| load_one(path, show_last, &cache, debug_level))
                .collect();

            // Persist this batch's fresh rows before sending it, from
            // references, so nothing here is cloned.
            save_conversations(
                ProviderKind::Codex,
                show_last,
                loaded.iter().filter_map(ConversationLoad::as_fresh),
            );

            let batch: Vec<Conversation> = loaded
                .into_iter()
                .map(ConversationLoad::into_conversation)
                .collect();
            if !batch.is_empty() {
                let _ = tx.send(LoaderMessage::Batch(batch));
            }
        }

        // Every directory has consumed the cache entries for files it saw;
        // whatever remains points at files that no longer exist. Pruning once
        // at the end keeps the same semantics as the sync path.
        let stale: Vec<PathBuf> = cache
            .into_inner()
            .map(|cache| cache.into_keys().collect())
            .unwrap_or_default();
        prune_conversations(ProviderKind::Codex, &stale);

        let _ = tx.send(LoaderMessage::Done);
    }
}

/// Load one transcript, preferring a fresh cache hit and consuming the cache
/// entry either way (so a file that merely failed to stat or parse is never
/// treated as deleted). Shared by the sync and streaming loaders.
fn load_one(
    path: PathBuf,
    show_last: bool,
    cache: &Mutex<HashMap<PathBuf, CachedFileConversation>>,
    debug_level: Option<DebugLevel>,
) -> Option<ConversationLoad> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();
    let metadata = fs::metadata(&path).ok();
    let modified = metadata
        .as_ref()
        .and_then(|metadata| metadata.modified().ok());
    let fingerprint = metadata.as_ref().map(fingerprint_from_metadata);

    // Always consume the cache entry for a file we've seen: whatever remains in
    // the map afterwards is treated as deleted and pruned, and a file that
    // merely failed to stat or parse isn't deleted.
    let cached_entry = cache.lock().ok().and_then(|mut cache| cache.remove(&path));
    if let (Some(fingerprint), Some(cached)) = (fingerprint, cached_entry)
        && let Some(conversation) = cached.into_conversation_if_fresh(fingerprint)
    {
        debug::debug(
            debug_level,
            &format!("Loaded Codex transcript {} from index", filename),
        );
        return Some(ConversationLoad::Cached(conversation));
    }

    match process_codex_transcript_file(path, show_last, modified, debug_level) {
        Ok(Some(conversation)) => {
            debug::debug(
                debug_level,
                &format!(
                    "Loaded Codex transcript {}: {}",
                    filename, conversation.preview
                ),
            );
            match fingerprint {
                Some(fingerprint) => Some(ConversationLoad::Fresh(conversation, fingerprint)),
                None => Some(ConversationLoad::Cached(conversation)),
            }
        }
        Ok(None) => None,
        Err(err) => {
            debug::warn(
                debug_level,
                &format!("Failed to process Codex transcript {}: {}", filename, err),
            );
            None
        }
    }
}

impl super::Provider for CodexProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Codex
    }

    fn name(&self) -> &str {
        "Codex"
    }

    fn load_conversations(
        &self,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) -> Result<Vec<Conversation>> {
        self.load_all_conversations(show_last, debug_level)
    }

    fn load_conversations_streaming(
        &self,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) -> Receiver<LoaderMessage> {
        let (tx, rx) = mpsc::channel();
        let codex_home = self.codex_home.clone();

        std::thread::spawn(move || {
            let provider = CodexProvider { codex_home };
            provider.stream_all_conversations(&tx, show_last, debug_level);
        });

        rx
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        // The viewer only needs the reconstructed entries; entries mode skips
        // the preview/full-text assembly the scan produces. show_last is
        // irrelevant here (it only steers the scan's preview direction).
        Ok(
            parse_codex_transcript_file(&conversation.path, ParseMode::Entries, false, None, None)?
                .entries,
        )
    }

    fn resume(&self, conversation: &Conversation, _default_args: &[String]) -> Result<()> {
        let mut command = Command::new("codex");
        command.arg("resume").arg(&conversation.id);
        // Legacy transcripts may record no cwd at all; codex can still resume those
        // by id, so only a recorded-but-missing directory is an error.
        match conversation
            .project_path
            .as_ref()
            .or(conversation.cwd.as_ref())
        {
            Some(path) if path.is_dir() => {
                command.current_dir(path);
            }
            Some(path) => {
                return Err(AppError::ClaudeExecutionError(format!(
                    "Project directory no longer exists: {}",
                    path.display()
                )));
            }
            None => {}
        }

        run_codex_command(command)
    }

    fn delete(&self, conversation: &Conversation) -> Result<()> {
        let mut command = Command::new("codex");
        command.arg("archive").arg(&conversation.id);
        if let Some(project_path) = conversation
            .project_path
            .as_ref()
            .or(conversation.cwd.as_ref())
            .filter(|path| path.is_dir())
        {
            command.current_dir(project_path);
        }

        // Capture output: this runs inside the live TUI (raw mode + alternate
        // screen), so the child must not inherit the terminal.
        let output = command
            .output()
            .map_err(|err| AppError::ClaudeExecutionError(err.to_string()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AppError::ClaudeExecutionError(format!(
                "codex archive exited with status {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        delete_conversation(ProviderKind::Codex, &conversation.path);
        Ok(())
    }
}

fn process_codex_transcript_file(
    path: PathBuf,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<Option<Conversation>> {
    // The startup listing only needs the conversation summary; scan mode skips
    // building the (discarded) LogEntry list entirely.
    Ok(
        parse_codex_transcript_file(&path, ParseMode::Scan, show_last, modified, debug_level)?
            .conversation,
    )
}

fn parse_codex_transcript_file(
    path: &Path,
    mode: ParseMode,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<ParsedCodexTranscript> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse_codex_transcript_reader(
        path.to_path_buf(),
        reader,
        mode,
        show_last,
        modified,
        debug_level,
    )
}

fn parse_codex_transcript_reader<R: BufRead>(
    path: PathBuf,
    reader: R,
    mode: ParseMode,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<ParsedCodexTranscript> {
    let mut state = CodexParseState::default();
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                state.parse_errors.push(ParseError {
                    line_number: line_idx + 1,
                    line_content: String::new(),
                    error_message: err.to_string(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        // Read the cheap head first. In scan mode, records the summary never
        // consumes (tool calls, tool outputs, reasoning) contribute only their
        // timestamp from here, so their large payloads are never parsed into a
        // Value. Everything the active mode does consume falls through to the
        // full parse below.
        match serde_json::from_str::<RecordHead>(&line) {
            Ok(head) => {
                if let Some(raw) = &head.timestamp {
                    record_raw_timestamp(&mut state, raw);
                }
                if head_needs_full_parse(&head, mode) {
                    process_line_body(
                        &mut state,
                        &line,
                        mode,
                        line_idx + 1,
                        debug_level,
                        &filename,
                    );
                }
            }
            Err(head_err) => {
                // The head parse is lenient about field types, so a failure here
                // means either malformed JSON or a shape the head can't model
                // (e.g. a non-object line, or a scalar `payload`). Only the
                // former was a parse error under the old Value path; a
                // valid-but-unhandled line was silently ignored (its `type`
                // lookup found nothing). Re-check with a full parse to keep that
                // distinction exact.
                match serde_json::from_str::<Value>(&line) {
                    Ok(value) => {
                        // The head never captured this line's timestamp; fold it
                        // in here so the bounds match the old Value-only path.
                        if let Some(raw) = value.get("timestamp").and_then(Value::as_str) {
                            record_raw_timestamp(&mut state, raw);
                        }
                        process_codex_record(&mut state, &value, mode, line_idx + 1);
                    }
                    Err(_) => record_parse_error(
                        &mut state,
                        line,
                        line_idx + 1,
                        head_err,
                        debug_level,
                        &filename,
                    ),
                }
            }
        }
    }

    finalize_timestamps(&mut state);

    let entries = std::mem::take(&mut state.entries);
    let conversation = match mode {
        ParseMode::Scan => build_codex_conversation(path, show_last, modified, state),
        // The viewer path builds no conversation; skip preview/full-text work.
        ParseMode::Entries => None,
    };
    Ok(ParsedCodexTranscript {
        conversation,
        entries,
    })
}

/// Whether the active mode needs the full `Value` parse for this record.
///
/// Scan mode consumes only session/turn metadata, messages (preview/counts) and
/// token counts; everything else contributes only its timestamp (already taken
/// from the head). Entries mode reconstructs the transcript, so it parses every
/// record that can yield a `LogEntry` — messages, tool calls, tool outputs,
/// web-search calls and reasoning — but still skips token counts, which produce
/// no entry.
fn head_needs_full_parse(head: &RecordHead, mode: ParseMode) -> bool {
    if head.is_legacy_session_metadata() {
        // Legacy metadata feeds id/cwd/model and, crucially for the entries
        // path, the fallback timestamp used to stamp entries, so both modes
        // parse it.
        return true;
    }
    match head.item_kind() {
        // Session/turn metadata populate id/cwd/model/metadata_timestamp, needed
        // by the scan summary and by the entries path for per-entry cwd/model
        // and timestamp fallback.
        Some("session_meta" | "turn_context") => true,
        Some("message") => true,
        // token_count only feeds the summary's total_tokens.
        Some("token_count") => mode == ParseMode::Scan,
        // These only ever produce a LogEntry, so the scan skips them and pays
        // nothing for their (often large) payloads.
        Some(
            "function_call"
            | "custom_tool_call"
            | "web_search_call"
            | "function_call_output"
            | "custom_tool_call_output"
            | "reasoning",
        ) => mode == ParseMode::Entries,
        _ => false,
    }
}

/// Full-parse one line and dispatch it. Only reached for records the active
/// mode actually consumes, so materializing the `Value` here is not wasted.
fn process_line_body(
    state: &mut CodexParseState,
    line: &str,
    mode: ParseMode,
    line_number: usize,
    debug_level: Option<DebugLevel>,
    filename: &str,
) {
    match serde_json::from_str::<Value>(line) {
        Ok(value) => process_codex_record(state, &value, mode, line_number),
        Err(err) => record_parse_error(
            state,
            line.to_string(),
            line_number,
            err,
            debug_level,
            filename,
        ),
    }
}

fn record_parse_error(
    state: &mut CodexParseState,
    line: String,
    line_number: usize,
    err: serde_json::Error,
    debug_level: Option<DebugLevel>,
    filename: &str,
) {
    debug::warn(
        debug_level,
        &format!(
            "Codex parse error in {} at line {}: {}",
            filename, line_number, err
        ),
    );
    state.parse_errors.push(ParseError {
        line_number,
        line_content: line,
        error_message: err.to_string(),
        context_before: Vec::new(),
        context_after: Vec::new(),
    });
}

/// Dispatch a fully parsed record to the handler the active mode wants.
///
/// Timestamp bounds are owned by the read loop (recorded cheaply from the head),
/// so this only re-derives the record's own timestamp for per-entry stamping,
/// and only when it is needed — the scan path never stamps entries. The `record`
/// here has already passed the mode's `head_needs_full_parse` gate (or arrived
/// through the malformed-head fallback), so no work below is wasted.
fn process_codex_record(
    state: &mut CodexParseState,
    record: &Value,
    mode: ParseMode,
    line_idx: usize,
) {
    // Metadata records populate id/cwd/model/metadata_timestamp. Both modes
    // consume these: the scan for the conversation summary, the entries path for
    // each entry's cwd/model and its timestamp fallback.
    match record.get("type").and_then(Value::as_str) {
        Some("session_meta") => {
            if let Some(payload) = record.get("payload") {
                parse_session_meta(state, payload);
            }
            return;
        }
        Some("turn_context") => {
            if let Some(payload) = record.get("payload") {
                parse_turn_context(state, payload);
            }
            return;
        }
        _ => {}
    }

    if is_legacy_session_metadata(record) {
        parse_legacy_session_meta(state, record);
        return;
    }

    let item = record.get("payload").unwrap_or(record);
    let kind = item.get("type").and_then(Value::as_str);

    // Only the entries path stamps individual entries, so only it pays for
    // parsing the record's timestamp per line.
    let timestamp = match mode {
        ParseMode::Entries => record_timestamp(record),
        ParseMode::Scan => None,
    };

    match (mode, kind) {
        // Messages feed both paths (preview/counts for the scan, an entry for
        // the viewer); process_message_item branches on the mode internally.
        (_, Some("message")) => process_message_item(state, item, mode, timestamp),
        (ParseMode::Scan, Some("token_count")) => parse_token_count(state, item),
        (ParseMode::Entries, Some("function_call") | Some("custom_tool_call")) => {
            process_tool_call_item(state, item, timestamp, line_idx)
        }
        (ParseMode::Entries, Some("web_search_call")) => {
            process_web_search_item(state, item, timestamp, line_idx)
        }
        (ParseMode::Entries, Some("function_call_output") | Some("custom_tool_call_output")) => {
            process_tool_output_item(state, item, timestamp, line_idx)
        }
        (ParseMode::Entries, Some("reasoning")) => process_reasoning_item(state, item, timestamp),
        _ => {}
    }
}

fn parse_session_meta(state: &mut CodexParseState, payload: &Value) {
    if state.id.is_none() {
        state.id = payload
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
    if state.cwd.is_none() {
        state.cwd = payload
            .get("cwd")
            .and_then(Value::as_str)
            .map(PathBuf::from);
    }
    if state.metadata_timestamp.is_none() {
        state.metadata_timestamp = payload
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_timestamp);
    }
}

fn parse_legacy_session_meta(state: &mut CodexParseState, record: &Value) {
    if state.id.is_none() {
        state.id = record
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
    if state.metadata_timestamp.is_none() {
        state.metadata_timestamp = record
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_timestamp);
    }
}

fn parse_turn_context(state: &mut CodexParseState, payload: &Value) {
    if state.cwd.is_none() {
        state.cwd = payload
            .get("cwd")
            .and_then(Value::as_str)
            .map(PathBuf::from);
    }
    if state.workspace_roots.is_empty()
        && let Some(roots) = payload.get("workspace_roots").and_then(Value::as_array)
    {
        state.workspace_roots = roots
            .iter()
            .filter_map(Value::as_str)
            .map(PathBuf::from)
            .collect();
    }
    if state.model.is_none() {
        state.model = payload
            .get("model")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
}

fn process_message_item(
    state: &mut CodexParseState,
    item: &Value,
    mode: ParseMode,
    timestamp: Option<DateTime<FixedOffset>>,
) {
    let role = item.get("role").and_then(Value::as_str).unwrap_or_default();
    if role != "user" && role != "assistant" {
        return;
    }

    let blocks = text_blocks_from_codex_content(item.get("content"), role == "user");
    if blocks.is_empty() {
        return;
    }

    // The text is needed either way: as the preview/full-text contribution in
    // scan mode, and as the same non-empty guard the entries path applied.
    let text = extract_text_from_blocks(&blocks);
    if text.trim().is_empty() {
        return;
    }

    match mode {
        // Scan builds no entries; it only needs the text and the message count.
        ParseMode::Scan => {
            state.all_parts.push(text);
            state.message_count += 1;
        }
        // The viewer wants the reconstructed entry, not the preview text.
        ParseMode::Entries => {
            let timestamp = timestamp_string(timestamp, state.metadata_timestamp);
            let uuid = item
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            match role {
                "user" => state.entries.push(LogEntry::User {
                    message: UserMessage {
                        content: UserContent::Blocks(blocks),
                    },
                    timestamp,
                    cwd: state
                        .cwd
                        .as_ref()
                        .map(|path| path.to_string_lossy().to_string()),
                }),
                "assistant" => state.entries.push(LogEntry::Assistant {
                    message: AssistantMessage {
                        content: blocks,
                        model: state.model.clone(),
                        usage: None,
                        id: uuid,
                    },
                    timestamp,
                }),
                _ => {}
            }
        }
    }
}

fn process_tool_call_item(
    state: &mut CodexParseState,
    item: &Value,
    timestamp: Option<DateTime<FixedOffset>>,
    line_idx: usize,
) {
    let name = item
        .get("name")
        .or_else(|| item.get("tool_name"))
        .and_then(Value::as_str)
        .unwrap_or("tool")
        .to_string();
    let input = parse_tool_input(item);
    push_tool_call_entry(state, item, name, input, timestamp, line_idx);
}

// web_search_call items carry the request in `action` ({type: search|open_page,
// query/queries/url}) instead of `arguments`; map them onto the web_search /
// web_fetch formatters.
fn process_web_search_item(
    state: &mut CodexParseState,
    item: &Value,
    timestamp: Option<DateTime<FixedOffset>>,
    line_idx: usize,
) {
    let action = item.get("action").unwrap_or(&Value::Null);
    let query = action
        .get("query")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            action
                .get("queries")
                .and_then(Value::as_array)
                .map(|queries| {
                    queries
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
        })
        .filter(|query| !query.is_empty());
    let url = action.get("url").and_then(Value::as_str);

    let (name, input) = match (query, url) {
        (Some(query), _) => ("web_search", serde_json::json!({ "query": query })),
        (None, Some(url)) => ("web_fetch", serde_json::json!({ "url": url })),
        (None, None) => ("web_search", action.clone()),
    };
    push_tool_call_entry(state, item, name.to_string(), input, timestamp, line_idx);
}

fn push_tool_call_entry(
    state: &mut CodexParseState,
    item: &Value,
    name: String,
    input: Value,
    timestamp: Option<DateTime<FixedOffset>>,
    line_idx: usize,
) {
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("codex-call-{}", line_idx));
    let timestamp = timestamp_string(timestamp, state.metadata_timestamp);

    state.entries.push(LogEntry::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolUse { id, name, input }],
            model: state.model.clone(),
            usage: None,
            id: None,
        },
        timestamp,
    });
}

fn process_tool_output_item(
    state: &mut CodexParseState,
    item: &Value,
    timestamp: Option<DateTime<FixedOffset>>,
    line_idx: usize,
) {
    let tool_use_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("codex-result-{}", line_idx));
    let content = item
        .get("output")
        .or_else(|| item.get("result"))
        .cloned()
        .or(Some(Value::Null));
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let is_error = item.get("is_error").and_then(Value::as_bool);
    let timestamp = timestamp_string(timestamp, state.metadata_timestamp);

    state.entries.push(LogEntry::User {
        message: UserMessage {
            content: UserContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                status,
            }]),
        },
        timestamp,
        cwd: state
            .cwd
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    });
}

fn process_reasoning_item(
    state: &mut CodexParseState,
    item: &Value,
    timestamp: Option<DateTime<FixedOffset>>,
) {
    let Some(summary) = item
        .get("summary")
        .and_then(text_from_codex_content)
        .filter(|summary| !summary.trim().is_empty())
    else {
        return;
    };
    let timestamp = timestamp_string(timestamp, state.metadata_timestamp);

    state.entries.push(LogEntry::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::Thinking { thinking: summary }],
            model: state.model.clone(),
            usage: None,
            id: None,
        },
        timestamp,
    });
}

fn parse_token_count(state: &mut CodexParseState, item: &Value) {
    if let Some(total_tokens) = item
        .pointer("/info/total_token_usage/total_tokens")
        .and_then(Value::as_u64)
    {
        state.total_tokens = total_tokens;
    }
}

fn build_codex_conversation(
    path: PathBuf,
    show_last: bool,
    modified: Option<SystemTime>,
    state: CodexParseState,
) -> Option<Conversation> {
    if state.all_parts.is_empty() {
        return None;
    }

    let preview = if show_last {
        state
            .all_parts
            .iter()
            .rev()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ... ")
    } else {
        state
            .all_parts
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ... ")
    };

    let timestamp = state
        .last_timestamp
        .or(state.metadata_timestamp)
        .map(|timestamp| timestamp.with_timezone(&Local))
        .or_else(|| modified.map(DateTime::<Local>::from))
        .unwrap_or_else(Local::now);
    let duration_minutes = match (state.first_timestamp, state.last_timestamp) {
        (Some(first), Some(last)) => {
            let minutes = last.signed_duration_since(first).num_minutes();
            (minutes > 0).then_some(minutes as u64)
        }
        _ => None,
    };
    let project_path = state
        .cwd
        .or_else(|| state.workspace_roots.into_iter().next());
    let project_name = project_path
        .as_ref()
        .map(|path| format_short_name_from_path(path));
    let id = state
        .id
        .unwrap_or_else(|| session_id_from_path(&path).unwrap_or_else(|| "unknown".to_string()));

    Some(Conversation {
        path,
        provider: ProviderKind::Codex,
        id,
        timestamp,
        preview: normalize_whitespace(&preview),
        full_text: normalize_whitespace(&state.all_parts.join(" ")),
        project_name,
        project_path: project_path.clone(),
        cwd: project_path,
        message_count: state.message_count,
        parse_errors: state.parse_errors,
        summary: None,
        model: state.model,
        total_tokens: state.total_tokens,
        duration_minutes,
        search_text_lower: None,
        search_topic_end: None,
    })
}

fn collect_session_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // file_type() does not follow symlinks, so symlinked directories are
            // never descended (guards against cycles); symlinked files still
            // resolve through path.is_file().
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                dirs.push(path);
            } else if path.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path);
            }
        }
    }

    files
}

/// Group session transcripts by the directory that directly contains them, one
/// group per leaf directory (`sessions/YYYY/MM/DD/`). The streaming loader sends
/// one batch per group so history paints progressively instead of only after
/// the whole tree is parsed.
///
/// Every group holds the files found *directly* in its directory, so the union
/// over all groups is exactly [`collect_session_files`]'s set — no file is
/// processed twice even when a directory holds both transcripts and
/// subdirectories. Groups are returned newest-directory-first (by path name, so
/// the date hierarchy sorts chronologically) to surface recent work first.
fn collect_session_batch_dirs(root: &Path) -> Vec<(PathBuf, Vec<PathBuf>)> {
    let mut groups: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        let mut files = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            // Same symlink handling as collect_session_files: never descend a
            // symlinked directory, but still accept symlinked files.
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                dirs.push(path);
            } else if path.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path);
            }
        }
        if !files.is_empty() {
            groups.push((dir, files));
        }
    }

    // Newest first: YYYY/MM/DD path strings sort chronologically.
    groups.sort_by(|(a, _), (b, _)| b.cmp(a));
    groups
}

fn resolve_codex_home(env_value: Option<OsString>, home: Option<PathBuf>) -> PathBuf {
    env_value
        // codex itself treats a set-but-empty CODEX_HOME as unset.
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| home.map(|home| home.join(".codex")))
        .unwrap_or_default()
}

fn text_blocks_from_codex_content(
    content: Option<&Value>,
    filter_metadata: bool,
) -> Vec<ContentBlock> {
    match content {
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                let text = block.get("text").and_then(Value::as_str)?;
                if filter_metadata && is_codex_metadata_text(text) {
                    return None;
                }
                Some(ContentBlock::Text {
                    text: text.to_string(),
                })
            })
            .collect(),
        Some(Value::String(text)) if !(filter_metadata && is_codex_metadata_text(text)) => {
            vec![ContentBlock::Text { text: text.clone() }]
        }
        _ => Vec::new(),
    }
}

fn text_from_codex_content(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let text = items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n");
            (!text.is_empty()).then_some(text)
        }
        Value::Object(_) => value
            .get("text")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

fn parse_tool_input(item: &Value) -> Value {
    let Some(input) = item.get("arguments").or_else(|| item.get("input")) else {
        return Value::Null;
    };

    match input {
        Value::String(text) => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        value => value.clone(),
    }
}

fn record_timestamp(record: &Value) -> Option<DateTime<FixedOffset>> {
    record
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(parse_timestamp)
}

fn parse_timestamp(value: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(value).ok()
}

/// Fold a record's raw timestamp string into the first/last bounds without a
/// per-record allocation or (for the last) a per-record parse. The first bound
/// is parsed eagerly only until it is set; the last is retained as a string and
/// parsed once at EOF.
fn record_raw_timestamp(state: &mut CodexParseState, raw: &str) {
    if state.first_timestamp.is_none() {
        state.first_timestamp = parse_timestamp(raw);
    }
    state.last_timestamp_raw.clear();
    state.last_timestamp_raw.push_str(raw);
    state.has_last_timestamp = true;
}

/// Parse the retained last-timestamp string exactly once. A malformed final
/// timestamp leaves `last_timestamp` unset, and the conversation summary falls
/// back through `metadata_timestamp`/file mtime just as it would for a record
/// that never carried a timestamp.
fn finalize_timestamps(state: &mut CodexParseState) {
    if state.has_last_timestamp {
        state.last_timestamp = parse_timestamp(&state.last_timestamp_raw);
    }
}

fn timestamp_string(
    timestamp: Option<DateTime<FixedOffset>>,
    fallback: Option<DateTime<FixedOffset>>,
) -> String {
    timestamp
        .or(fallback)
        .map(|ts| ts.to_rfc3339())
        .unwrap_or_default()
}

fn is_legacy_session_metadata(record: &Value) -> bool {
    record.get("id").is_some()
        && record.get("timestamp").is_some()
        && record.get("type").is_none()
        && record.get("payload").is_none()
}

// Matches synthetic user-role records codex injects (instructions, environment
// snapshots, Esc-interrupt markers). These always START with their marker;
// matching anywhere in the text would eat genuine user messages that quote one.
fn is_codex_metadata_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("# AGENTS.md instructions for ")
        || trimmed.starts_with("<environment_context>")
        || trimmed.starts_with("<turn_aborted>")
}

fn session_id_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    // A UUID suffix is pure ASCII, so a non-char-boundary cut point (multi-byte
    // character straddling it) can never hold one; slicing there would panic.
    if stem.len() >= 36 && stem.is_char_boundary(stem.len() - 36) {
        let candidate = &stem[stem.len() - 36..];
        if is_uuid_like(candidate) {
            return Some(candidate.to_string());
        }
    }
    Some(stem.to_string())
}

fn is_uuid_like(value: &str) -> bool {
    value.len() == 36 && value.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-')
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(unix)]
fn run_codex_command(mut command: Command) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let err = command.exec();
    Err(AppError::ClaudeExecutionError(err.to_string()))
}

#[cfg(not(unix))]
fn run_codex_command(mut command: Command) -> Result<()> {
    let status = command
        .status()
        .map_err(|err| AppError::ClaudeExecutionError(err.to_string()))?;

    if !status.success() {
        return Err(AppError::ClaudeExecutionError(format!(
            "codex CLI exited with status {}",
            status
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude::{extract_text_from_assistant, extract_text_from_user};
    use std::io::Cursor;

    const TEST_PATH: &str =
        "rollout-2026-06-10T17-58-01-019eb42f-a4e2-75e0-b7ad-2268948a559c.jsonl";

    fn scan(content: &str, show_last: bool) -> ParsedCodexTranscript {
        parse_codex_transcript_reader(
            PathBuf::from(TEST_PATH),
            Cursor::new(content),
            ParseMode::Scan,
            show_last,
            None,
            None,
        )
        .unwrap()
    }

    fn entries(content: &str) -> Vec<LogEntry> {
        parse_codex_transcript_reader(
            PathBuf::from(TEST_PATH),
            Cursor::new(content),
            ParseMode::Entries,
            false,
            None,
            None,
        )
        .unwrap()
        .entries
    }

    /// Run both passes over the same transcript and combine them, so existing
    /// tests can keep asserting on the conversation summary (scan) and the
    /// reconstructed entries (entries) from a single call.
    fn parse(content: &str, show_last: bool) -> ParsedCodexTranscript {
        let scanned = scan(content, show_last);
        ParsedCodexTranscript {
            conversation: scanned.conversation,
            entries: entries(content),
        }
    }

    #[test]
    fn parses_current_codex_transcript() {
        let content = [
            r#"{"timestamp":"2026-06-11T00:58:16.810Z","type":"session_meta","payload":{"id":"019eb42f-a4e2-75e0-b7ad-2268948a559c","timestamp":"2026-06-11T00:58:01.875Z","cwd":"/tmp/project","source":"cli"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:16.812Z","type":"turn_context","payload":{"cwd":"/tmp/project","workspace_roots":["/tmp/project"],"model":"gpt-5.5"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:16.813Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\nnoise\n</environment_context>"},{"type":"input_text","text":"Build Codex support"}]}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.514Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Working on it"}],"phase":"commentary"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.515Z","type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"cargo test\"}","call_id":"call_1"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.714Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"ok"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.714Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"total_tokens":1234}}}}"#,
        ]
        .join("\n");

        let parsed = parse(&content, false);
        let conversation = parsed.conversation.unwrap();

        assert_eq!(conversation.provider, ProviderKind::Codex);
        assert_eq!(conversation.id, "019eb42f-a4e2-75e0-b7ad-2268948a559c");
        assert_eq!(
            conversation.project_path,
            Some(PathBuf::from("/tmp/project"))
        );
        assert_eq!(conversation.model.as_deref(), Some("gpt-5.5"));
        assert_eq!(conversation.total_tokens, 1234);
        assert_eq!(conversation.message_count, 2);
        assert!(conversation.preview.contains("Build Codex support"));
        assert!(!conversation.preview.contains("environment_context"));
        assert_eq!(parsed.entries.len(), 4);
    }

    #[test]
    fn parses_legacy_codex_transcript() {
        let content = [
            r#"{"id":"a4bcd2da-17d7-430a-b2e2-3fc80e1ac920","timestamp":"2025-08-31T21:41:00.914Z","instructions":"test"}"#,
            r#"{"record_type":"response_item"}"#,
            r#"{"type":"message","id":"msg_1","role":"user","content":[{"type":"input_text","text":"Hello Codex"}]}"#,
            r#"{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Thinking summary"}],"encrypted_content":"..."}"#,
            r#"{"type":"message","id":"msg_2","role":"assistant","content":[{"type":"output_text","text":"Hello back"}]}"#,
            r#"{"type":"function_call","id":"fc_1","name":"shell","arguments":"{\"command\":\"pwd\"}","call_id":"call_1"}"#,
            r#"{"type":"function_call_output","call_id":"call_1","output":"/tmp"}"#,
        ]
        .join("\n");

        let parsed = parse(&content, true);
        let conversation = parsed.conversation.unwrap();

        assert_eq!(conversation.id, "a4bcd2da-17d7-430a-b2e2-3fc80e1ac920");
        assert!(conversation.preview.starts_with("Hello back"));
        assert_eq!(conversation.message_count, 2);
        assert_eq!(parsed.entries.len(), 5);
    }

    #[test]
    fn extracts_session_id_from_rollout_filename() {
        let path =
            PathBuf::from("rollout-2026-06-10T17-58-01-019eb42f-a4e2-75e0-b7ad-2268948a559c.jsonl");

        assert_eq!(
            session_id_from_path(&path).as_deref(),
            Some("019eb42f-a4e2-75e0-b7ad-2268948a559c")
        );
    }

    // Assert two conversation summaries are field-for-field identical for every
    // value the cached row persists. Behavioral identity of the scan output is
    // the hard requirement: a cached row must never change meaning.
    fn assert_same_summary(a: &Conversation, b: &Conversation) {
        assert_eq!(a.id, b.id, "id");
        assert_eq!(a.preview, b.preview, "preview");
        assert_eq!(a.full_text, b.full_text, "full_text");
        assert_eq!(a.message_count, b.message_count, "message_count");
        assert_eq!(a.model, b.model, "model");
        assert_eq!(a.total_tokens, b.total_tokens, "total_tokens");
        assert_eq!(a.timestamp, b.timestamp, "timestamp");
        assert_eq!(a.project_path, b.project_path, "project_path");
        assert_eq!(a.cwd, b.cwd, "cwd");
        assert_eq!(a.duration_minutes, b.duration_minutes, "duration_minutes");
    }

    #[test]
    fn scan_ignores_giant_tool_output_but_keeps_identical_summary() {
        // A multi-megabyte function_call_output is the bulk of a transcript.
        // Scan mode must produce exactly the summary it would if that payload
        // were empty — it contributes only its timestamp bound, never its body.
        let giant = "X".repeat(4 * 1024 * 1024);
        let with_giant = [
            r#"{"timestamp":"2026-06-11T00:58:16.810Z","type":"session_meta","payload":{"id":"019eb42f-a4e2-75e0-b7ad-2268948a559c","timestamp":"2026-06-11T00:58:01.875Z","cwd":"/tmp/project","source":"cli"}}"#.to_string(),
            r#"{"timestamp":"2026-06-11T00:58:16.812Z","type":"turn_context","payload":{"cwd":"/tmp/project","workspace_roots":["/tmp/project"],"model":"gpt-5.5"}}"#.to_string(),
            r#"{"timestamp":"2026-06-11T00:58:16.813Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build Codex support"}]}}"#.to_string(),
            r#"{"timestamp":"2026-06-11T00:58:27.514Z","type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"cargo test\"}","call_id":"call_1"}}"#.to_string(),
            format!(
                r#"{{"timestamp":"2026-06-11T00:58:27.714Z","type":"response_item","payload":{{"type":"function_call_output","call_id":"call_1","output":"{giant}"}}}}"#
            ),
            r#"{"timestamp":"2026-06-11T00:58:30.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}"#.to_string(),
            r#"{"timestamp":"2026-06-11T00:58:30.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"total_tokens":1234}}}}"#.to_string(),
        ]
        .join("\n");

        // Golden reference: the same transcript with the tool-output body emptied
        // (its timestamp is unchanged, so it still contributes the same bound).
        let with_empty = with_giant.replace(&giant, "");

        let scanned_giant = scan(&with_giant, false).conversation.unwrap();
        let scanned_empty = scan(&with_empty, false).conversation.unwrap();

        assert_same_summary(&scanned_giant, &scanned_empty);

        // Concrete expected values, so the golden compare can't pass by both
        // sides being wrong the same way.
        assert_eq!(scanned_giant.id, "019eb42f-a4e2-75e0-b7ad-2268948a559c");
        assert_eq!(scanned_giant.preview, "Build Codex support ... Done");
        assert_eq!(scanned_giant.full_text, "Build Codex support Done");
        assert_eq!(scanned_giant.message_count, 2);
        assert_eq!(scanned_giant.model.as_deref(), Some("gpt-5.5"));
        assert_eq!(scanned_giant.total_tokens, 1234);
        // The giant tool output never becomes preview/full-text.
        assert!(!scanned_giant.full_text.contains('X'));
        assert!(!scanned_giant.preview.contains('X'));

        // Scan mode builds no entries at all — that work is exclusive to the
        // viewer path.
        assert!(scan(&with_giant, false).entries.is_empty());
    }

    #[test]
    fn read_entries_reconstructs_transcript_and_scan_builds_no_conversation() {
        // Same transcript, exercised through the entries (viewer) path. The
        // reconstructed LogEntry list must be unchanged: user msg, assistant
        // msg, the tool-call, and the tool-result carrying the full output.
        let content = [
            r#"{"timestamp":"2026-06-11T00:58:16.810Z","type":"session_meta","payload":{"id":"019eb42f-a4e2-75e0-b7ad-2268948a559c","timestamp":"2026-06-11T00:58:01.875Z","cwd":"/tmp/project","source":"cli"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:16.812Z","type":"turn_context","payload":{"cwd":"/tmp/project","workspace_roots":["/tmp/project"],"model":"gpt-5.5"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:16.813Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build Codex support"}]}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.514Z","type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"cargo test\"}","call_id":"call_1"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:27.714Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"BUILD OK"}}"#,
            r#"{"timestamp":"2026-06-11T00:58:30.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}}"#,
        ]
        .join("\n");

        let log = entries(&content);
        assert_eq!(log.len(), 4);

        // 0: user message
        match &log[0] {
            LogEntry::User { message, cwd, .. } => {
                assert_eq!(cwd.as_deref(), Some("/tmp/project"));
                assert_eq!(extract_text_from_user(message), "Build Codex support");
            }
            other => panic!("expected user entry, got {other:?}"),
        }
        // 1: assistant tool-call, stamped with the turn's model
        match &log[1] {
            LogEntry::Assistant { message, .. } => match message.content.first() {
                Some(ContentBlock::ToolUse { name, .. }) => assert_eq!(name, "exec_command"),
                other => panic!("expected tool_use, got {other:?}"),
            },
            other => panic!("expected assistant entry, got {other:?}"),
        }
        // 2: tool result carries the full output body verbatim
        match &log[2] {
            LogEntry::User { message, .. } => match &message.content {
                UserContent::Blocks(blocks) => match blocks.first() {
                    Some(ContentBlock::ToolResult { content, .. }) => {
                        assert_eq!(content.as_ref().and_then(Value::as_str), Some("BUILD OK"));
                    }
                    other => panic!("expected tool_result, got {other:?}"),
                },
                other => panic!("expected blocks, got {other:?}"),
            },
            other => panic!("expected user entry, got {other:?}"),
        }
        // 3: assistant final message
        match &log[3] {
            LogEntry::Assistant { message, .. } => {
                assert_eq!(extract_text_from_assistant(message), "Done");
                assert_eq!(message.model.as_deref(), Some("gpt-5.5"));
            }
            other => panic!("expected assistant entry, got {other:?}"),
        }

        // Mode contract: the entries pass builds no conversation summary.
        let entries_pass = parse_codex_transcript_reader(
            PathBuf::from(TEST_PATH),
            Cursor::new(&content),
            ParseMode::Entries,
            false,
            None,
            None,
        )
        .unwrap();
        assert!(entries_pass.conversation.is_none());
    }

    #[test]
    fn filters_turn_aborted_but_keeps_users_quoting_metadata_tags() {
        let content = [
            r#"{"timestamp":"2026-06-11T00:58:16.813Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<turn_aborted>\nThe user interrupted the previous turn on purpose."}]}}"#,
            r#"{"timestamp":"2026-06-11T00:58:20.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"why does codex inject this?\n<environment_context>\nnoise\n</environment_context>"}]}}"#,
        ]
        .join("\n");

        let parsed = parse(&content, false);
        let conversation = parsed.conversation.unwrap();

        assert!(!conversation.full_text.contains("turn_aborted"));
        assert!(
            conversation
                .full_text
                .contains("why does codex inject this?")
        );
        assert_eq!(conversation.message_count, 1);
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn maps_web_search_calls_to_named_tool_entries() {
        let content = [
            r#"{"timestamp":"2026-06-11T00:58:16.813Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Search the docs"}]}}"#,
            r#"{"timestamp":"2026-06-11T00:58:17.000Z","type":"response_item","payload":{"type":"web_search_call","status":"completed","action":{"type":"search","query":"codex config reference"}}}"#,
            r#"{"timestamp":"2026-06-11T00:58:18.000Z","type":"response_item","payload":{"type":"web_search_call","status":"completed","action":{"type":"open_page","url":"https://example.com/docs"}}}"#,
        ]
        .join("\n");

        let parsed = parse(&content, false);
        let tool_uses: Vec<(&str, &Value)> = parsed
            .entries
            .iter()
            .filter_map(|entry| match entry {
                LogEntry::Assistant { message, .. } => match message.content.first() {
                    Some(ContentBlock::ToolUse { name, input, .. }) => Some((name.as_str(), input)),
                    _ => None,
                },
                _ => None,
            })
            .collect();

        assert_eq!(tool_uses.len(), 2);
        assert_eq!(tool_uses[0].0, "web_search");
        assert_eq!(
            tool_uses[0].1.get("query").and_then(Value::as_str),
            Some("codex config reference")
        );
        assert_eq!(tool_uses[1].0, "web_fetch");
        assert_eq!(
            tool_uses[1].1.get("url").and_then(Value::as_str),
            Some("https://example.com/docs")
        );
    }

    #[test]
    fn session_id_handles_non_ascii_filenames() {
        // 37-byte stem whose 36-bytes-from-the-end cut point lands inside 'é'.
        let straddling = PathBuf::from(format!("é{}.jsonl", "a".repeat(35)));
        let expected = format!("é{}", "a".repeat(35));
        assert_eq!(
            session_id_from_path(&straddling).as_deref(),
            Some(expected.as_str())
        );

        let uuid_after_multibyte_prefix = PathBuf::from(
            "日本語-rollout-2026-06-10T17-58-01-019eb42f-a4e2-75e0-b7ad-2268948a559c.jsonl",
        );
        assert_eq!(
            session_id_from_path(&uuid_after_multibyte_prefix).as_deref(),
            Some("019eb42f-a4e2-75e0-b7ad-2268948a559c")
        );
    }

    #[test]
    fn resolves_codex_home_treating_empty_env_as_unset() {
        let home = Some(PathBuf::from("/home/user"));

        assert_eq!(
            resolve_codex_home(Some(OsString::from("/custom/codex")), home.clone()),
            PathBuf::from("/custom/codex")
        );
        assert_eq!(
            resolve_codex_home(Some(OsString::new()), home.clone()),
            PathBuf::from("/home/user/.codex")
        );
        assert_eq!(
            resolve_codex_home(None, home),
            PathBuf::from("/home/user/.codex")
        );
        assert_eq!(resolve_codex_home(None, None), PathBuf::new());

        let provider = CodexProvider {
            codex_home: PathBuf::new(),
        };
        assert!(provider.sessions_root().is_none());
    }

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new(name: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mnemonai-codex-{}-{}-{}",
                name,
                std::process::id(),
                unique
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn batches_sessions_per_day_directory_newest_first() {
        let tmp = TestTempDir::new("batch-dirs");
        let root = &tmp.path;

        // sessions/2026/06/{10,11}/, plus a stray transcript directly under a
        // month directory, plus a non-jsonl file that must be ignored.
        let day_10 = root.join("2026/06/10");
        let day_11 = root.join("2026/06/11");
        let month = root.join("2026/06");
        fs::create_dir_all(&day_10).unwrap();
        fs::create_dir_all(&day_11).unwrap();
        fs::write(day_10.join("rollout-a.jsonl"), b"{}").unwrap();
        fs::write(day_10.join("rollout-b.jsonl"), b"{}").unwrap();
        fs::write(day_11.join("rollout-c.jsonl"), b"{}").unwrap();
        fs::write(month.join("stray.jsonl"), b"{}").unwrap();
        fs::write(day_11.join("notes.txt"), b"ignore me").unwrap();

        let groups = collect_session_batch_dirs(root);

        // One group per directory that directly holds a transcript, newest
        // (lexicographically greatest path) first: .../06/11, .../06/10, .../06.
        let dirs: Vec<&Path> = groups.iter().map(|(dir, _)| dir.as_path()).collect();
        assert_eq!(
            dirs,
            vec![day_11.as_path(), day_10.as_path(), month.as_path()]
        );

        // Each group holds exactly its own directory's jsonl files, and their
        // union is the whole corpus — no file counted twice, .txt excluded.
        let counts: Vec<usize> = groups.iter().map(|(_, files)| files.len()).collect();
        assert_eq!(counts, vec![1, 2, 1]);
        let total: usize = counts.iter().sum();
        assert_eq!(total, collect_session_files(root).len());
    }
}
