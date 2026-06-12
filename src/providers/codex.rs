use crate::claude::{AssistantMessage, ContentBlock, LogEntry, UserContent, UserMessage};
use crate::cli::DebugLevel;
use crate::conversation_index::{
    SourceFingerprint, delete_conversation, fingerprint_from_metadata, load_provider_cache,
    prune_conversations, save_conversations,
};
use crate::debug;
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, ParseError, ProviderKind, format_short_name_from_path,
};
use chrono::{DateTime, FixedOffset, Local};
use rayon::prelude::*;
use serde_json::Value;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver};
use std::time::SystemTime;

pub struct CodexProvider {
    codex_home: PathBuf,
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
    first_timestamp: Option<DateTime<FixedOffset>>,
    last_timestamp: Option<DateTime<FixedOffset>>,
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
            .filter_map(|path| {
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

                // Always consume the cache entry for a file we've seen: whatever
                // remains in the map afterwards is treated as deleted and pruned,
                // and a file that merely failed to stat or parse isn't deleted.
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
                            Some(fingerprint) => {
                                Some(ConversationLoad::Fresh(conversation, fingerprint))
                            }
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
            })
            .collect();

        save_conversations(
            ProviderKind::Codex,
            show_last,
            loaded.iter().filter_map(|loaded| match loaded {
                ConversationLoad::Fresh(conversation, fingerprint) => {
                    Some((conversation, *fingerprint))
                }
                ConversationLoad::Cached(_) => None,
            }),
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
        for (idx, conversation) in conversations.iter_mut().enumerate() {
            conversation.index = idx;
        }
        Ok(conversations)
    }
}

impl super::Provider for CodexProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Codex
    }

    fn name(&self) -> &str {
        "Codex"
    }

    fn detect(&self) -> bool {
        self.sessions_root().is_some_and(|root| root.exists())
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
            match provider.load_all_conversations(show_last, debug_level) {
                Ok(conversations) => {
                    if !conversations.is_empty() {
                        let _ = tx.send(LoaderMessage::Batch(conversations));
                    }
                }
                Err(_) => {
                    let _ = tx.send(LoaderMessage::ProjectError);
                }
            }
            let _ = tx.send(LoaderMessage::Done);
        });

        rx
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        Ok(parse_codex_transcript_file(&conversation.path, false, None, None)?.entries)
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
    Ok(parse_codex_transcript_file(&path, show_last, modified, debug_level)?.conversation)
}

fn parse_codex_transcript_file(
    path: &Path,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<ParsedCodexTranscript> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse_codex_transcript_reader(path.to_path_buf(), reader, show_last, modified, debug_level)
}

fn parse_codex_transcript_reader<R: BufRead>(
    path: PathBuf,
    reader: R,
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

        match serde_json::from_str::<Value>(&line) {
            Ok(value) => process_codex_record(&mut state, &value, line_idx + 1),
            Err(err) => {
                debug::warn(
                    debug_level,
                    &format!(
                        "Codex parse error in {} at line {}: {}",
                        filename,
                        line_idx + 1,
                        err
                    ),
                );
                state.parse_errors.push(ParseError {
                    line_number: line_idx + 1,
                    line_content: line,
                    error_message: err.to_string(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
            }
        }
    }

    let entries = std::mem::take(&mut state.entries);
    let conversation = build_codex_conversation(path, show_last, modified, state);
    Ok(ParsedCodexTranscript {
        conversation,
        entries,
    })
}

fn process_codex_record(state: &mut CodexParseState, record: &Value, line_idx: usize) {
    let timestamp = record_timestamp(record);
    if let Some(timestamp) = timestamp {
        update_timestamp_bounds(state, timestamp);
    }

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
    match item.get("type").and_then(Value::as_str) {
        Some("message") => process_message_item(state, item, timestamp),
        Some("function_call") | Some("custom_tool_call") => {
            process_tool_call_item(state, item, timestamp, line_idx)
        }
        Some("web_search_call") => process_web_search_item(state, item, timestamp, line_idx),
        Some("function_call_output") | Some("custom_tool_call_output") => {
            process_tool_output_item(state, item, timestamp, line_idx)
        }
        Some("reasoning") => process_reasoning_item(state, item, timestamp, line_idx),
        Some("token_count") => parse_token_count(state, item),
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

    let text = extract_text_from_content_blocks(&blocks);
    if text.trim().is_empty() {
        return;
    }

    let timestamp = timestamp_string(timestamp, state.metadata_timestamp);
    let uuid = item
        .get("id")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    match role {
        "user" => state.entries.push(LogEntry::User {
            message: UserMessage {
                role: "user".to_string(),
                content: UserContent::Blocks(blocks),
            },
            timestamp,
            uuid,
            cwd: state
                .cwd
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
        }),
        "assistant" => state.entries.push(LogEntry::Assistant {
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: blocks,
                model: state.model.clone(),
                usage: None,
                id: uuid,
            },
            timestamp,
            uuid: None,
        }),
        _ => {}
    }

    state.all_parts.push(text);
    state.message_count += 1;
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
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse { id, name, input }],
            model: state.model.clone(),
            usage: None,
            id: None,
        },
        timestamp,
        uuid: None,
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
    let timestamp = timestamp_string(timestamp, state.metadata_timestamp);

    state.entries.push(LogEntry::User {
        message: UserMessage {
            role: "user".to_string(),
            content: UserContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id,
                content,
            }]),
        },
        timestamp,
        uuid: None,
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
    line_idx: usize,
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
            role: "assistant".to_string(),
            content: vec![ContentBlock::Thinking {
                thinking: summary,
                signature: format!("codex-thinking-{}", line_idx),
            }],
            model: state.model.clone(),
            usage: None,
            id: None,
        },
        timestamp,
        uuid: None,
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
        index: 0,
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

fn extract_text_from_content_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
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

fn update_timestamp_bounds(state: &mut CodexParseState, timestamp: DateTime<FixedOffset>) {
    if state.first_timestamp.is_none() {
        state.first_timestamp = Some(timestamp);
    }
    state.last_timestamp = Some(timestamp);
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
    use std::io::Cursor;

    fn parse(content: &str, show_last: bool) -> ParsedCodexTranscript {
        parse_codex_transcript_reader(
            PathBuf::from("rollout-2026-06-10T17-58-01-019eb42f-a4e2-75e0-b7ad-2268948a559c.jsonl"),
            Cursor::new(content),
            show_last,
            None,
            None,
        )
        .unwrap()
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
}
