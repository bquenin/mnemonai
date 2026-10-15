//! Cursor Agent sessions driven over the Agent Client Protocol (ACP).
//!
//! In ACP mode (`cursor-agent acp`, which is how T3 Code and other ACP clients
//! run Cursor) the CLI writes no `agent-transcripts` JSONL. Each session lives
//! in `~/.cursor/acp-sessions/<session-id>/` as a `meta.json` (workspace `cwd`
//! and an optional `title`) plus a content-addressed SQLite `store.db`:
//!
//! - `meta` table: one row (`key = '0'`) of hex-encoded JSON with the session
//!   `name`, `createdAt` (epoch ms) and `latestRootBlobId`. The row also carries
//!   `blobEncryptionKey`, which [`StoreMeta`] deliberately never deserializes.
//! - `blobs` table: `id` is the hex sha256 of `data`. Message blobs are JSON in
//!   the Vercel AI SDK message shape; tree blobs are protobuf. A root tree's
//!   field 1 lists the conversation's message ids in order, and field 16 embeds
//!   one tree per subagent.
//!
//! Every turn writes a new root and older roots stay behind as prefix
//! snapshots, so only `latestRootBlobId` is followed.

use super::{normalize_whitespace, previews_from_parts, system_time_to_millis};
use crate::claude::{
    AssistantMessage, ContentBlock, LogEntry, UserContent, UserMessage, extract_text_from_blocks,
};
use crate::cli::DebugLevel;
use crate::conversation_index::SourceFingerprint;
use crate::debug;
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, ParseError, PreviewPair, ProviderKind, format_short_name_from_path,
};
use chrono::{DateTime, FixedOffset, Local, NaiveDateTime, TimeZone};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::{self, Metadata};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const STORE_DB: &str = "store.db";
const STORE_WAL: &str = "store.db-wal";
const SESSIONS_DIR: &str = "acp-sessions";

const META_JSON: &str = "meta.json";

/// A session directory that holds a `store.db`, with the metadata captured
/// during the directory walk.
pub(super) struct AcpSession {
    pub(super) id: String,
    pub(super) db_path: PathBuf,
    /// Last write to `store.db` or its WAL: the session's last activity.
    pub(super) modified: SystemTime,
    /// Covers `store.db`, its WAL (a live session appends to the WAL long
    /// before a checkpoint touches the main file) and `meta.json` (title and
    /// workspace).
    pub(super) fingerprint: SourceFingerprint,
}

/// The result of listing the ACP root.
#[derive(Default)]
pub(super) struct AcpScan {
    pub(super) sessions: Vec<AcpSession>,
    /// Paths whose cache rows must survive pruning because they could not be
    /// read this time: a session directory whose `store.db` failed to stat, or
    /// the whole root when it (or one of its entries) could not be listed.
    pub(super) keep_under: Vec<PathBuf>,
}

/// Whether `path` is an ACP session's `store.db` rather than a JSONL transcript.
pub fn is_session_store(path: &Path) -> bool {
    path.file_name().is_some_and(|name| name == STORE_DB)
        && path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .is_some_and(|name| name == SESSIONS_DIR)
}

/// The session directory owning a `store.db` path.
pub(super) fn session_dir(db_path: &Path) -> Option<&Path> {
    is_session_store(db_path)
        .then(|| db_path.parent())
        .flatten()
}

/// List the sessions under `root`, newest first. Directories without a
/// `store.db` (just a `meta.json`) are sessions the client opened and never
/// used, so they are skipped. A missing root is an empty listing. Anything
/// that could not be read is reported in [`AcpScan::keep_under`] so the caller
/// keeps those cache rows instead of pruning them.
pub(super) fn collect_sessions(root: &Path) -> AcpScan {
    let mut scan = AcpScan::default();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == ErrorKind::NotFound => return scan,
        Err(_) => {
            scan.keep_under.push(root.to_path_buf());
            return scan;
        }
    };

    for entry in entries {
        let Ok(entry) = entry else {
            // An unreadable entry has no name to protect; protect them all.
            scan.keep_under.push(root.to_path_buf());
            continue;
        };
        let dir = entry.path();
        let db_path = dir.join(STORE_DB);
        let db = match fs::metadata(&db_path) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(_) => {
                scan.keep_under.push(dir);
                continue;
            }
        };
        let Some(id) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let wal = fs::metadata(dir.join(STORE_WAL)).ok();
        let meta_json = fs::metadata(dir.join(META_JSON)).ok();

        let mtime = |metadata: &Metadata| metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let modified = wal.iter().map(mtime).fold(mtime(&db), SystemTime::max);
        let fingerprint_modified = meta_json.iter().map(mtime).fold(modified, SystemTime::max);
        let size = [Some(&db), wal.as_ref(), meta_json.as_ref()]
            .into_iter()
            .flatten()
            .map(Metadata::len)
            .sum::<u64>();
        scan.sessions.push(AcpSession {
            id,
            db_path,
            modified,
            fingerprint: SourceFingerprint {
                modified_millis: system_time_to_millis(fingerprint_modified),
                size: size.min(i64::MAX as u64) as i64,
            },
        });
    }

    scan.sessions
        .sort_by_key(|session| std::cmp::Reverse(session.modified));
    scan
}

/// Build the listing entry for one session. Returns `None` only for a session
/// that decoded cleanly but holds no conversation. A session whose store or
/// messages cannot be decoded (for example after a `cursor-agent` format
/// change) is still listed with its title, workspace, time and parse errors.
pub(super) fn process_session(
    session: &AcpSession,
    show_last: bool,
    debug_level: Option<DebugLevel>,
) -> Option<(Conversation, PreviewPair)> {
    let loaded = load_session(&session.db_path);
    if let Some(err) = &loaded.store_error {
        debug::warn(
            debug_level,
            &format!(
                "Cursor ACP session {}: {}; listing its metadata only",
                session.id, err
            ),
        );
    }

    let cwd = loaded.meta_file.cwd.clone();
    let cwd_string = cwd
        .as_deref()
        .map(|path| path.to_string_lossy().into_owned());
    let converted = convert(&loaded, cwd_string.as_deref());
    if loaded.store_error.is_none()
        && loaded.parse_errors.is_empty()
        && converted.message_count == 0
    {
        return None;
    }

    let title = loaded.title();
    let mut parts = converted.parts;
    if parts.is_empty() {
        parts.push(
            title
                .clone()
                .unwrap_or_else(|| "Cursor ACP session".to_string()),
        );
    }
    let previews = previews_from_parts(&parts);
    let preview = previews.select(show_last).to_string();
    let full_text = normalize_whitespace(&parts.join(" "));

    let started = loaded
        .store
        .created_at
        .and_then(DateTime::from_timestamp_millis);
    let last_activity = converted
        .last_activity
        .unwrap_or_else(|| DateTime::<Local>::from(session.modified).fixed_offset());
    let duration_minutes = started.and_then(|started| {
        let minutes = last_activity.signed_duration_since(started).num_minutes();
        (minutes > 0).then_some(minutes as u64)
    });

    let mut parse_errors = loaded.parse_errors;
    if let Some(err) = loaded.store_error {
        parse_errors.push(ParseError {
            line_number: 0,
            line_content: String::new(),
            error_message: err,
            context_before: Vec::new(),
            context_after: Vec::new(),
        });
    }

    Some((
        Conversation {
            path: session.db_path.clone(),
            provider: ProviderKind::CursorAgent,
            id: session.id.clone(),
            timestamp: DateTime::<Local>::from(session.modified),
            preview,
            full_text,
            project_name: cwd.as_deref().map(format_short_name_from_path),
            project_path: cwd.clone(),
            cwd,
            message_count: converted.message_count,
            parse_errors,
            summary: title,
            model: converted.model,
            total_tokens: 0,
            duration_minutes,
        },
        previews,
    ))
}

/// Read a session's entries for the viewer and exports: the main conversation
/// in order, with each subagent's messages placed after the tool result that
/// returned it.
pub(super) fn read_entries(db_path: &Path) -> Result<Vec<LogEntry>> {
    let loaded = load_session(db_path);
    if let Some(err) = &loaded.store_error {
        return Err(AppError::ClaudeExecutionError(format!(
            "Failed to read Cursor ACP session {}: {}",
            db_path.display(),
            err
        )));
    }
    let cwd = loaded
        .meta_file
        .cwd
        .as_deref()
        .map(|path| path.to_string_lossy().into_owned());
    Ok(convert(&loaded, cwd.as_deref()).entries)
}

#[derive(Debug, Default, Deserialize)]
struct SessionMetaFile {
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    title: Option<String>,
}

/// The decoded `meta` row. Lists only the fields used here so that
/// `blobEncryptionKey` is never held, logged, cached or exported.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoreMeta {
    #[serde(default)]
    latest_root_blob_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct AcpMessage {
    role: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    content: AcpContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AcpContent {
    Text(String),
    Parts(Vec<AcpPart>),
}

impl Default for AcpContent {
    fn default() -> Self {
        AcpContent::Parts(Vec::new())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum AcpPart {
    Text {
        #[serde(default)]
        text: String,
    },
    Reasoning {
        #[serde(default)]
        text: String,
        #[serde(default, rename = "providerOptions")]
        provider_options: Option<Value>,
    },
    ToolCall {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(default, alias = "input")]
        args: Value,
    },
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(default, alias = "output")]
        result: Value,
        #[serde(default, rename = "isError")]
        is_error: Option<bool>,
    },
    /// `redacted-reasoning`, images and future part types.
    #[serde(other)]
    Other,
}

/// A subagent's conversation, embedded in the main root tree.
struct Subagent {
    id: String,
    created_millis: Option<i64>,
    updated_millis: Option<i64>,
    messages: Vec<AcpMessage>,
}

#[derive(Default)]
struct LoadedSession {
    meta_file: SessionMetaFile,
    store: StoreMeta,
    main: Vec<AcpMessage>,
    subagents: Vec<Subagent>,
    /// Why the store could not be decoded; the session is then listed from
    /// `meta.json` alone.
    store_error: Option<String>,
    parse_errors: Vec<ParseError>,
}

impl LoadedSession {
    /// `meta.json` title, then the store's session name. `None` lets the
    /// listing fall back to the first user message.
    fn title(&self) -> Option<String> {
        [self.meta_file.title.as_deref(), self.store.name.as_deref()]
            .into_iter()
            .flatten()
            .map(str::trim)
            .find(|title| !title.is_empty())
            .map(str::to_string)
    }
}

fn load_session(db_path: &Path) -> LoadedSession {
    let mut session = LoadedSession {
        meta_file: db_path
            .parent()
            .and_then(|dir| fs::read(dir.join("meta.json")).ok())
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default(),
        ..LoadedSession::default()
    };
    if let Err(err) = load_store(db_path, &mut session) {
        session.store_error = Some(err);
    }
    session
}

fn load_store(db_path: &Path, session: &mut LoadedSession) -> std::result::Result<(), String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|err| format!("cannot open store: {err}"))?;

    let meta_hex: String = conn
        .query_row("SELECT value FROM meta WHERE key = '0'", [], |row| {
            row.get(0)
        })
        .map_err(|err| format!("cannot read store metadata: {err}"))?;
    let meta_json =
        decode_hex(meta_hex.trim()).ok_or_else(|| "store metadata is not hex".to_string())?;
    session.store = serde_json::from_slice(&meta_json)
        .map_err(|err| format!("cannot parse store metadata: {err}"))?;

    let mut blobs = conn
        .prepare("SELECT data FROM blobs WHERE id = ?1")
        .map_err(|err| format!("cannot read blobs: {err}"))?;
    let mut blob = |id: &str| -> Option<Vec<u8>> {
        blobs
            .query_row([id], |row| row.get::<_, Vec<u8>>(0))
            .optional()
            .ok()
            .flatten()
    };

    let root_id = session
        .store
        .latest_root_blob_id
        .clone()
        .ok_or_else(|| "store metadata has no latestRootBlobId".to_string())?;
    let root_bytes = blob(&root_id).ok_or_else(|| "root tree blob is missing".to_string())?;
    let root = decode_tree(&root_bytes).ok_or_else(|| "cannot decode root tree".to_string())?;

    let mut parse_errors = Vec::new();
    let mut read_messages = |ids: &[String]| -> Vec<AcpMessage> {
        ids.iter()
            .enumerate()
            .filter_map(|(index, id)| {
                let parsed = match blob(id) {
                    Some(data) => {
                        serde_json::from_slice::<AcpMessage>(&data).map_err(|err| err.to_string())
                    }
                    None => Err("message blob is missing".to_string()),
                };
                parsed
                    .map_err(|error_message| {
                        parse_errors.push(ParseError {
                            line_number: index + 1,
                            line_content: format!("blob {id}"),
                            error_message,
                            context_before: Vec::new(),
                            context_after: Vec::new(),
                        });
                    })
                    .ok()
            })
            .collect()
    };

    session.main = read_messages(&root.messages);
    session.subagents = root
        .subagents
        .into_iter()
        .map(|subagent| Subagent {
            messages: read_messages(&subagent.messages),
            id: subagent.id,
            created_millis: subagent.created_millis,
            updated_millis: subagent.updated_millis,
        })
        .collect();
    session.parse_errors = parse_errors;
    Ok(())
}

#[derive(Default)]
struct Converted {
    entries: Vec<LogEntry>,
    /// Preview/search text of the main conversation, one per message.
    parts: Vec<String>,
    message_count: usize,
    last_activity: Option<DateTime<FixedOffset>>,
    model: Option<String>,
}

fn convert(session: &LoadedSession, cwd: Option<&str>) -> Converted {
    let mut out = Converted::default();
    let mut unplaced: Vec<&Subagent> = session.subagents.iter().collect();

    for message in &session.main {
        match message.role.as_str() {
            "user" => {
                let text = user_text(&message.content);
                let timestamp = parse_user_timestamp(&text);
                out.note_activity(timestamp);
                // The environment block (`<user_info>`, rules, skills) and
                // compaction summaries are injected by the client, not typed.
                let is_meta = is_injected_context(&text);
                if !is_meta {
                    out.message_count += 1;
                    let preview = normalize_whitespace(&text);
                    if !preview.is_empty() {
                        out.parts.push(preview);
                    }
                }
                out.entries.push(LogEntry::User {
                    message: UserMessage {
                        content: UserContent::Blocks(vec![ContentBlock::Text { text }]),
                    },
                    timestamp: timestamp.map(|ts| ts.to_rfc3339()).unwrap_or_default(),
                    cwd: cwd.map(str::to_string),
                    is_meta,
                });
            }
            "assistant" => {
                let model = model_name(&message.content);
                if out.model.is_none() {
                    out.model.clone_from(&model);
                }
                let blocks = assistant_blocks(&message.content);
                if blocks.is_empty() {
                    continue;
                }
                out.message_count += 1;
                if let Some(text) = preview_text(&blocks) {
                    out.parts.push(text);
                }
                out.entries.push(LogEntry::Assistant {
                    message: AssistantMessage {
                        content: blocks,
                        model,
                        usage: None,
                        id: message.id.clone(),
                    },
                    timestamp: String::new(),
                });
            }
            "tool" => {
                let blocks = tool_result_blocks(&message.content);
                if blocks.is_empty() {
                    continue;
                }
                out.message_count += 1;
                let spawned = spawned_agent_ids(&blocks);
                out.entries.push(LogEntry::User {
                    message: UserMessage {
                        content: UserContent::Blocks(blocks),
                    },
                    timestamp: String::new(),
                    cwd: cwd.map(str::to_string),
                    is_meta: false,
                });
                for agent_id in spawned {
                    if let Some(index) = unplaced.iter().position(|sub| sub.id == agent_id) {
                        out.push_subagent(unplaced.remove(index));
                    }
                }
            }
            // `system` prompts are not part of the conversation.
            _ => {}
        }
    }

    // Subagents whose result never came back (still running, aborted) go last.
    unplaced.sort_by_key(|sub| sub.created_millis);
    for subagent in unplaced {
        out.push_subagent(subagent);
    }
    out
}

impl Converted {
    fn note_activity(&mut self, timestamp: Option<DateTime<FixedOffset>>) {
        if let Some(ts) = timestamp
            && self.last_activity.is_none_or(|last| ts > last)
        {
            self.last_activity = Some(ts);
        }
    }

    /// Append a subagent's messages as `agent_progress` entries, the shape the
    /// viewer and `show --json` already render for Claude Code subagents.
    fn push_subagent(&mut self, subagent: &Subagent) {
        let updated = subagent
            .updated_millis
            .and_then(DateTime::from_timestamp_millis)
            .map(|ts| ts.fixed_offset());
        self.note_activity(updated);

        for message in &subagent.messages {
            let (message_type, blocks) = match message.role.as_str() {
                "user" => {
                    let text = user_text(&message.content);
                    if text.trim().is_empty() {
                        continue;
                    }
                    ("user", vec![ContentBlock::Text { text }])
                }
                "assistant" => ("assistant", assistant_blocks(&message.content)),
                "tool" => ("user", tool_result_blocks(&message.content)),
                _ => continue,
            };
            if blocks.is_empty() {
                continue;
            }
            self.entries.push(LogEntry::Progress {
                data: json!({
                    "type": "agent_progress",
                    "agentId": subagent.id,
                    "message": {
                        "type": message_type,
                        "message": {
                            "content": blocks.iter().map(block_json).collect::<Vec<_>>(),
                        },
                    },
                }),
            });
        }
    }
}

fn user_text(content: &AcpContent) -> String {
    match content {
        AcpContent::Text(text) => text.clone(),
        AcpContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                AcpPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

fn is_injected_context(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("<user_info>") || text.starts_with("<conversation_summary>")
}

fn assistant_blocks(content: &AcpContent) -> Vec<ContentBlock> {
    let parts = match content {
        AcpContent::Text(text) if text.trim().is_empty() => return Vec::new(),
        AcpContent::Text(text) => return vec![ContentBlock::Text { text: text.clone() }],
        AcpContent::Parts(parts) => parts,
    };
    parts
        .iter()
        .filter_map(|part| match part {
            AcpPart::Text { text } if !text.trim().is_empty() => {
                Some(ContentBlock::Text { text: text.clone() })
            }
            // Reasoning text is usually empty (the content sits in an opaque
            // signature); only show it when there is something to read.
            AcpPart::Reasoning { text, .. } if !text.trim().is_empty() => {
                Some(ContentBlock::Thinking {
                    thinking: text.clone(),
                })
            }
            AcpPart::ToolCall {
                tool_call_id,
                tool_name,
                args,
            } => Some(ContentBlock::ToolUse {
                id: tool_call_id.clone(),
                name: tool_name.clone(),
                input: args.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn tool_result_blocks(content: &AcpContent) -> Vec<ContentBlock> {
    let AcpContent::Parts(parts) = content else {
        return Vec::new();
    };
    parts
        .iter()
        .filter_map(|part| match part {
            AcpPart::ToolResult {
                tool_call_id,
                result,
                is_error,
            } => Some(ContentBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: (!result.is_null()).then(|| result.clone()),
                is_error: *is_error,
                status: None,
            }),
            _ => None,
        })
        .collect()
}

/// Model from the first reasoning part's `providerOptions.cursor.modelName`.
fn model_name(content: &AcpContent) -> Option<String> {
    let AcpContent::Parts(parts) = content else {
        return None;
    };
    parts.iter().find_map(|part| match part {
        AcpPart::Reasoning {
            provider_options: Some(options),
            ..
        } => options
            .pointer("/cursor/modelName")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    })
}

/// Text for previews and search, or a tool summary for tool-only messages.
fn preview_text(blocks: &[ContentBlock]) -> Option<String> {
    let text = normalize_whitespace(&extract_text_from_blocks(blocks));
    if !text.is_empty() {
        return Some(text);
    }
    let summary = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { name, .. } => Some(format!("Tool: {name}")),
            ContentBlock::Thinking { .. } => Some("Thinking".to_string()),
            _ => None,
        })
        .take(3)
        .collect::<Vec<_>>()
        .join(" ... ");
    (!summary.is_empty()).then_some(summary)
}

/// Subagent ids a tool result reports. The subagent tool ends its output with
/// `Agent ID: <uuid>`.
fn spawned_agent_ids(blocks: &[ContentBlock]) -> Vec<String> {
    const MARKER: &str = "Agent ID: ";
    let mut ids = Vec::new();
    for block in blocks {
        let ContentBlock::ToolResult {
            content: Some(content),
            ..
        } = block
        else {
            continue;
        };
        let text = match content {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        for (index, _) in text.match_indices(MARKER) {
            let id: String = text[index + MARKER.len()..]
                .chars()
                .take_while(|c| c.is_ascii_hexdigit() || *c == '-')
                .collect();
            if !id.is_empty() {
                ids.push(id);
            }
        }
    }
    ids
}

/// Serialize a block in the JSON shape `ContentBlock` deserializes from, for
/// `agent_progress` payloads.
fn block_json(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
        ContentBlock::ToolUse { id, name, input } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            status,
        } => json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
            "status": status,
        }),
        ContentBlock::Thinking { thinking } => json!({ "type": "thinking", "thinking": thinking }),
        ContentBlock::Image { source } => json!({ "type": "image", "source": source }),
    }
}

/// The `<timestamp>` header Cursor prepends to each user query.
fn parse_user_timestamp(text: &str) -> Option<DateTime<FixedOffset>> {
    const OPEN: &str = "<timestamp>";
    let start = text.find(OPEN)? + OPEN.len();
    let end = start + text[start..].find("</timestamp>")?;
    parse_cursor_timestamp(text[start..end].trim())
}

/// Parse Cursor's `Thursday, Oct 1, 2026, 9:55 AM (UTC-7)`.
fn parse_cursor_timestamp(value: &str) -> Option<DateTime<FixedOffset>> {
    let (local, zone) = value.rsplit_once(" (")?;
    let zone = zone.strip_suffix(')')?.strip_prefix("UTC")?;
    let offset_seconds = if zone.is_empty() {
        0
    } else {
        let sign = match zone.as_bytes()[0] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        let (hours, minutes) = zone[1..].split_once(':').unwrap_or((&zone[1..], "0"));
        sign * (hours.parse::<i32>().ok()? * 3600 + minutes.parse::<i32>().ok()? * 60)
    };
    let offset = FixedOffset::east_opt(offset_seconds)?;
    let naive = NaiveDateTime::parse_from_str(local, "%A, %b %d, %Y, %I:%M %p").ok()?;
    offset.from_local_datetime(&naive).single()
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(hex.get(index..index + 2)?, 16).ok())
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// A decoded conversation tree: ordered message blob ids plus subagent trees.
#[derive(Default)]
struct Tree {
    messages: Vec<String>,
    subagents: Vec<SubagentTree>,
}

struct SubagentTree {
    id: String,
    created_millis: Option<i64>,
    updated_millis: Option<i64>,
    messages: Vec<String>,
}

/// Decode a tree blob. There is no published `.proto`; the fields used here
/// were mapped by hand:
///
/// - 1 (repeated bytes): raw 32-byte sha256 of each message blob, in order.
/// - 16 (repeated message): a subagent, `{1: id, 2: {1: embedded tree,
///   2: created ms, 3: updated ms}}`.
fn decode_tree(buf: &[u8]) -> Option<Tree> {
    let mut tree = Tree::default();
    for (field, value) in decode_fields(buf)? {
        match (field, value) {
            (1, Wire::Bytes(id)) if id.len() == 32 => tree.messages.push(encode_hex(id)),
            (16, Wire::Bytes(subagent)) => {
                if let Some(subagent) = decode_subagent(subagent) {
                    tree.subagents.push(subagent);
                }
            }
            _ => {}
        }
    }
    Some(tree)
}

fn decode_subagent(buf: &[u8]) -> Option<SubagentTree> {
    let mut id = None;
    let mut body = None;
    for (field, value) in decode_fields(buf)? {
        match (field, value) {
            (1, Wire::Bytes(bytes)) => id = std::str::from_utf8(bytes).ok().map(str::to_string),
            (2, Wire::Bytes(bytes)) => body = Some(bytes),
            _ => {}
        }
    }

    let mut subagent = SubagentTree {
        id: id?,
        created_millis: None,
        updated_millis: None,
        messages: Vec::new(),
    };
    for (field, value) in decode_fields(body?)? {
        match (field, value) {
            (1, Wire::Bytes(tree)) => subagent.messages = decode_tree(tree)?.messages,
            (2, Wire::Varint(millis)) => subagent.created_millis = i64::try_from(millis).ok(),
            (3, Wire::Varint(millis)) => subagent.updated_millis = i64::try_from(millis).ok(),
            _ => {}
        }
    }
    Some(subagent)
}

enum Wire<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// Split a protobuf message into `(field number, value)` pairs. Returns `None`
/// on any malformed or truncated input instead of panicking.
fn decode_fields(buf: &[u8]) -> Option<Vec<(u64, Wire<'_>)>> {
    let mut pos = 0;
    let mut fields = Vec::new();
    while pos < buf.len() {
        let key = read_varint(buf, &mut pos)?;
        let value = match key & 7 {
            0 => Wire::Varint(read_varint(buf, &mut pos)?),
            1 => {
                pos = pos.checked_add(8).filter(|end| *end <= buf.len())?;
                Wire::Fixed
            }
            2 => {
                let len = usize::try_from(read_varint(buf, &mut pos)?).ok()?;
                let end = pos.checked_add(len).filter(|end| *end <= buf.len())?;
                let bytes = &buf[pos..end];
                pos = end;
                Wire::Bytes(bytes)
            }
            5 => {
                pos = pos.checked_add(4).filter(|end| *end <= buf.len())?;
                Wire::Fixed
            }
            _ => return None,
        };
        fields.push((key >> 3, value));
    }
    Some(fields)
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude::parse_agent_progress;
    use std::time::UNIX_EPOCH;

    const SECRET_KEY: &str = "deadbeefcafebabe0123456789abcdef";

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mnemonai-cursor-acp-{}-{}-{}",
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

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn bytes_field(field: u64, bytes: &[u8]) -> Vec<u8> {
        let mut out = varint(field << 3 | 2);
        out.extend(varint(bytes.len() as u64));
        out.extend_from_slice(bytes);
        out
    }

    fn varint_field(field: u64, value: u64) -> Vec<u8> {
        let mut out = varint(field << 3);
        out.extend(varint(value));
        out
    }

    /// A fake 32-byte blob id; only needs to be unique within a fixture.
    fn blob_id(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn tree(message_ids: &[[u8; 32]], extra: &[Vec<u8>]) -> Vec<u8> {
        let mut out = bytes_field(9, b"file:///work/space");
        for id in message_ids {
            out.extend(bytes_field(1, id));
        }
        out.extend(varint_field(10, 1));
        for field in extra {
            out.extend_from_slice(field);
        }
        out
    }

    fn subagent_field(id: &str, message_ids: &[[u8; 32]], created: u64, updated: u64) -> Vec<u8> {
        let mut body = bytes_field(1, &tree(message_ids, &[]));
        body.extend(varint_field(2, created));
        body.extend(varint_field(3, updated));
        body.extend(bytes_field(5, b"composer-2.5-fast"));
        let mut sub = bytes_field(1, id.as_bytes());
        sub.extend(bytes_field(2, &body));
        bytes_field(16, &sub)
    }

    struct Fixture {
        _root: TestTempDir,
        db_path: PathBuf,
        conn: Connection,
    }

    impl Fixture {
        fn new(name: &str, meta_json: &str) -> Self {
            let root = TestTempDir::new(name);
            let dir = root.path.join(SESSIONS_DIR).join("session-1");
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("meta.json"), meta_json).unwrap();
            let db_path = dir.join(STORE_DB);
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB);
                 CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);",
            )
            .unwrap();
            Self {
                _root: root,
                db_path,
                conn,
            }
        }

        fn blob(&self, id: &[u8; 32], data: &[u8]) {
            self.conn
                .execute(
                    "INSERT INTO blobs (id, data) VALUES (?1, ?2)",
                    rusqlite::params![encode_hex(id), data],
                )
                .unwrap();
        }

        fn message(&self, id: u8, json: Value) -> [u8; 32] {
            let blob = blob_id(id);
            self.blob(&blob, json.to_string().as_bytes());
            blob
        }

        fn store_meta(&self, root: &[u8; 32], name: Option<&str>) {
            let mut meta = json!({
                "agentId": "session-1",
                "latestRootBlobId": encode_hex(root),
                "mode": "default",
                "createdAt": 1_791_319_515_013_i64,
                "blobEncryptionKey": SECRET_KEY,
            });
            if let Some(name) = name {
                meta["name"] = json!(name);
            }
            self.conn
                .execute(
                    "INSERT INTO meta (key, value) VALUES ('0', ?1)",
                    [encode_hex(meta.to_string().as_bytes())],
                )
                .unwrap();
        }

        fn session(&self) -> AcpSession {
            let root = self.db_path.parent().unwrap().parent().unwrap();
            let mut scan = collect_sessions(root);
            assert!(scan.keep_under.is_empty());
            scan.sessions.pop().expect("fixture session")
        }
    }

    /// A session with an env block, two turns (one spawning a subagent), an
    /// older prefix-snapshot root, and an unplaced second subagent.
    fn standard_fixture(name: &str, meta_json: &str, store_name: Option<&str>) -> Fixture {
        let fx = Fixture::new(name, meta_json);
        let system = fx.message(1, json!({"role": "system", "content": "You are an agent"}));
        let env = fx.message(
            2,
            json!({"role": "user", "content": "<user_info>\nOS: linux\n</user_info>"}),
        );
        let query = fx.message(
            3,
            json!({"role": "user", "content": [{"type": "text", "text":
                "<timestamp>Thursday, Oct 1, 2026, 9:55 AM (UTC-7)</timestamp>\n<user_query>\nreview the PR\n</user_query>"}]}),
        );
        let call = fx.message(
            4,
            json!({"role": "assistant", "id": "a1", "content": [
                {"type": "reasoning", "text": "", "signature": "opaque",
                 "providerOptions": {"cursor": {"modelName": "grok-4.7-high"}}},
                {"type": "text", "text": "Spawning a checker"},
                {"type": "tool-call", "toolCallId": "call-1", "toolName": "CallDynamicTool",
                 "args": {"prompt": "check it"}}
            ]}),
        );
        let result = fx.message(
            5,
            json!({"role": "tool", "id": "t1", "content": [
                {"type": "tool-result", "toolCallId": "call-1", "toolName": "CallDynamicTool",
                 "result": "Verdict: holds.\n\nAgent ID: 11111111-2222-3333-4444-555555555555 (can be used to resume)",
                 "experimental_content": []}
            ]}),
        );
        let follow_up = fx.message(
            6,
            json!({"role": "user", "content":
                "<timestamp>Thursday, Oct 1, 2026, 10:12 AM (UTC-7)</timestamp>\n<user_query>\nthanks\n</user_query>"}),
        );
        let answer = fx.message(
            7,
            json!({"role": "assistant", "id": "a2", "content": [{"type": "text", "text": "Done"}]}),
        );

        let sub_system = fx.message(20, json!({"role": "system", "content": "sub system"}));
        let sub_user = fx.message(
            21,
            json!({"role": "user", "content": "<system_reminder>You are a subagent</system_reminder> check it"}),
        );
        let sub_answer = fx.message(
            22,
            json!({"role": "assistant", "content": [{"type": "text", "text": "subagent says holds"}]}),
        );
        let orphan_user = fx.message(30, json!({"role": "user", "content": "orphan task"}));

        let main = [system, env, query, call, result, follow_up, answer];
        let subagents = [
            subagent_field(
                "99999999-0000-0000-0000-000000000000",
                &[orphan_user],
                1_791_320_000_000,
                1_791_320_100_000,
            ),
            subagent_field(
                "11111111-2222-3333-4444-555555555555",
                &[sub_system, sub_user, sub_answer],
                1_791_319_600_000,
                1_791_319_700_000,
            ),
        ];

        // An older snapshot whose field-1 list is a prefix of the latest root.
        let old_root = blob_id(100);
        fx.blob(&old_root, &tree(&main[..3], &[]));
        let latest_root = blob_id(101);
        fx.blob(&latest_root, &tree(&main, &subagents));
        fx.store_meta(&latest_root, store_name);
        fx
    }

    fn texts(entries: &[LogEntry]) -> Vec<String> {
        entries
            .iter()
            .map(|entry| match entry {
                LogEntry::User {
                    message, is_meta, ..
                } => {
                    let prefix = if *is_meta { "meta" } else { "user" };
                    match &message.content {
                        UserContent::Blocks(blocks) => match blocks.first() {
                            Some(ContentBlock::ToolResult { tool_use_id, .. }) => {
                                format!("result:{tool_use_id}")
                            }
                            _ => format!("{prefix}:{}", extract_text_from_blocks(blocks)),
                        },
                        UserContent::String(text) => format!("{prefix}:{text}"),
                    }
                }
                LogEntry::Assistant { message, .. } => {
                    format!("assistant:{}", extract_text_from_blocks(&message.content))
                }
                LogEntry::Progress { data } => {
                    let progress = parse_agent_progress(data).expect("agent progress");
                    let crate::claude::AgentContent::Blocks(blocks) =
                        &progress.message.message.content;
                    format!(
                        "agent {}:{}:{}",
                        &progress.agent_id[..8],
                        progress.message.message_type,
                        extract_text_from_blocks(blocks)
                    )
                }
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn read_entries_follows_latest_root_and_places_subagents() {
        let fx = standard_fixture("order", r#"{"schemaVersion":1,"cwd":"/work/space"}"#, None);

        let entries = read_entries(&fx.db_path).unwrap();

        assert_eq!(
            texts(&entries),
            vec![
                "meta:<user_info>\nOS: linux\n</user_info>".to_string(),
                "user:<timestamp>Thursday, Oct 1, 2026, 9:55 AM (UTC-7)</timestamp>\n<user_query>\nreview the PR\n</user_query>".to_string(),
                "assistant:Spawning a checker".to_string(),
                "result:call-1".to_string(),
                "agent 11111111:user:<system_reminder>You are a subagent</system_reminder> check it".to_string(),
                "agent 11111111:assistant:subagent says holds".to_string(),
                "user:<timestamp>Thursday, Oct 1, 2026, 10:12 AM (UTC-7)</timestamp>\n<user_query>\nthanks\n</user_query>".to_string(),
                "assistant:Done".to_string(),
                "agent 99999999:user:orphan task".to_string(),
            ]
        );

        match &entries[2] {
            LogEntry::Assistant { message, .. } => {
                assert_eq!(message.model.as_deref(), Some("grok-4.7-high"));
                assert!(matches!(
                    &message.content[1],
                    ContentBlock::ToolUse { id, name, input }
                        if id == "call-1" && name == "CallDynamicTool" && input["prompt"] == "check it"
                ));
            }
            other => panic!("expected assistant entry, got {other:?}"),
        }
        match &entries[1] {
            LogEntry::User { timestamp, cwd, .. } => {
                assert_eq!(timestamp, "2026-10-01T09:55:00-07:00");
                assert_eq!(cwd.as_deref(), Some("/work/space"));
            }
            other => panic!("expected user entry, got {other:?}"),
        }
    }

    #[test]
    fn process_session_builds_metadata_and_hides_the_key() {
        let fx = standard_fixture(
            "metadata",
            r#"{"schemaVersion":1,"cwd":"/work/space","title":"PR Reviewer"}"#,
            Some("Store Name"),
        );

        let (conversation, previews) = process_session(&fx.session(), false, None).unwrap();

        assert_eq!(conversation.id, "session-1");
        assert_eq!(conversation.provider, ProviderKind::CursorAgent);
        assert_eq!(conversation.path, fx.db_path);
        assert_eq!(conversation.cwd, Some(PathBuf::from("/work/space")));
        assert_eq!(
            conversation.project_path,
            Some(PathBuf::from("/work/space"))
        );
        assert_eq!(conversation.summary.as_deref(), Some("PR Reviewer"));
        assert_eq!(conversation.model.as_deref(), Some("grok-4.7-high"));
        // Two user queries, two assistant turns, one tool result.
        assert_eq!(conversation.message_count, 5);
        assert!(conversation.parse_errors.is_empty());
        // The env block is not the preview; the real query is.
        assert!(
            previews.first.contains("review the PR"),
            "{}",
            previews.first
        );
        assert!(!conversation.full_text.contains("user_info"));
        // createdAt to the orphan subagent's last update.
        assert_eq!(conversation.duration_minutes, Some(9));

        let listed = format!(
            "{} {} {:?} {:?} {} {}",
            conversation.preview,
            conversation.full_text,
            conversation.summary,
            conversation.parse_errors,
            previews.first,
            previews.last
        );
        assert!(!listed.contains(SECRET_KEY));
        let entries = format!("{:?}", read_entries(&fx.db_path).unwrap());
        assert!(!entries.contains(SECRET_KEY));
    }

    #[test]
    fn title_falls_back_to_store_name_then_preview() {
        let fx = standard_fixture("title-name", r#"{"cwd":"/w"}"#, Some("Store Name"));
        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.summary.as_deref(), Some("Store Name"));

        let fx = standard_fixture("title-none", r#"{"cwd":"/w","title":"  "}"#, None);
        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.summary, None);
        assert!(conversation.preview.contains("review the PR"));
    }

    #[test]
    fn garbage_root_lists_metadata_without_panicking() {
        let fx = Fixture::new("garbage", r#"{"cwd":"/work/space","title":"Broken"}"#);
        let root = blob_id(1);
        // Wire type 7 is invalid; also truncated.
        fx.blob(&root, &[0x0f, 0xff, 0xff]);
        fx.store_meta(&root, None);

        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.summary.as_deref(), Some("Broken"));
        assert_eq!(conversation.preview, "Broken");
        assert_eq!(conversation.message_count, 0);
        assert_eq!(conversation.parse_errors.len(), 1);
        assert!(read_entries(&fx.db_path).is_err());
    }

    #[test]
    fn missing_root_lists_metadata() {
        let fx = Fixture::new("missing-root", r#"{"cwd":"/work/space"}"#);
        fx.store_meta(&blob_id(9), Some("Named"));

        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.summary.as_deref(), Some("Named"));
        assert_eq!(conversation.cwd, Some(PathBuf::from("/work/space")));
    }

    #[test]
    fn decoded_session_without_messages_is_skipped() {
        let fx = Fixture::new("empty", r#"{"cwd":"/work/space"}"#);
        let system = fx.message(1, json!({"role": "system", "content": "prompt"}));
        let root = blob_id(2);
        fx.blob(&root, &tree(&[system], &[]));
        fx.store_meta(&root, None);

        assert!(process_session(&fx.session(), false, None).is_none());
    }

    #[test]
    fn unparseable_message_is_a_parse_error_not_a_failure() {
        let fx = Fixture::new("bad-message", r#"{"cwd":"/w"}"#);
        let good = fx.message(1, json!({"role": "user", "content": "hello"}));
        let bad = blob_id(2);
        fx.blob(&bad, b"{not json");
        let root = blob_id(3);
        fx.blob(&root, &tree(&[good, bad, blob_id(4)], &[]));
        fx.store_meta(&root, None);

        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.message_count, 1);
        assert_eq!(conversation.parse_errors.len(), 2);
        assert!(!format!("{:?}", conversation.parse_errors).contains(SECRET_KEY));
    }

    #[test]
    fn session_with_only_damaged_messages_is_still_listed() {
        let fx = Fixture::new("all-damaged", r#"{"cwd":"/w","title":"Damaged"}"#);
        let bad = blob_id(1);
        fx.blob(&bad, b"{not json");
        let root = blob_id(2);
        fx.blob(&root, &tree(&[bad, blob_id(3)], &[]));
        fx.store_meta(&root, None);

        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();
        assert_eq!(conversation.summary.as_deref(), Some("Damaged"));
        assert_eq!(conversation.message_count, 0);
        assert_eq!(conversation.parse_errors.len(), 2);
    }

    #[test]
    fn meta_json_changes_the_fingerprint() {
        let fx = standard_fixture("meta-fingerprint", r#"{"cwd":"/w","title":"Old"}"#, None);
        let before = fx.session().fingerprint;

        let meta_path = fx.db_path.parent().unwrap().join(META_JSON);
        fs::write(&meta_path, r#"{"cwd":"/w","title":"Renamed session"}"#).unwrap();

        assert_ne!(fx.session().fingerprint, before);
    }

    #[test]
    fn provider_reads_refuses_resume_and_deletes_acp_sessions() {
        use crate::providers::Provider;

        let fx = standard_fixture("provider", r#"{"cwd":"/work/space"}"#, None);
        let session_dir = fx.db_path.parent().unwrap().to_path_buf();
        let provider = super::super::CursorAgentProvider {
            projects_root: fx._root.path.join("projects"),
            acp_root: session_dir.parent().unwrap().to_path_buf(),
        };
        let (conversation, _) = process_session(&fx.session(), false, None).unwrap();

        assert_eq!(provider.read_entries(&conversation).unwrap().len(), 9);
        assert!(provider.resume(&conversation, &[]).is_err());
        provider.delete(&conversation).unwrap();
        assert!(!session_dir.exists());
    }

    #[test]
    fn collect_sessions_skips_stub_dirs() {
        let root = TestTempDir::new("collect");
        let stub = root.path.join("stub");
        fs::create_dir_all(&stub).unwrap();
        fs::write(stub.join("meta.json"), r#"{"schemaVersion":1,"cwd":"/w"}"#).unwrap();
        let real = root.path.join("real");
        fs::create_dir_all(&real).unwrap();
        fs::write(real.join(STORE_DB), b"db").unwrap();
        fs::write(real.join(STORE_WAL), b"wal!").unwrap();
        fs::write(real.join(META_JSON), b"{}").unwrap();

        let scan = collect_sessions(&root.path);
        assert!(scan.keep_under.is_empty());
        assert_eq!(scan.sessions.len(), 1);
        assert_eq!(scan.sessions[0].id, "real");
        assert_eq!(scan.sessions[0].fingerprint.size, 8);

        let missing = collect_sessions(&root.path.join("missing"));
        assert!(missing.sessions.is_empty());
        assert!(missing.keep_under.is_empty());
    }

    #[test]
    fn collect_sessions_protects_what_it_cannot_read() {
        let root = TestTempDir::new("unreadable");
        let not_a_dir = root.path.join("file");
        fs::write(&not_a_dir, b"").unwrap();
        let scan = collect_sessions(&not_a_dir);
        assert!(scan.sessions.is_empty());
        assert_eq!(scan.keep_under, vec![not_a_dir]);

        #[cfg(unix)]
        {
            // A self-referencing symlink fails to stat with ELOOP, not NotFound.
            let looped = root.path.join("acp").join("looped");
            fs::create_dir_all(&looped).unwrap();
            std::os::unix::fs::symlink(looped.join(STORE_DB), looped.join(STORE_DB)).unwrap();
            let scan = collect_sessions(&root.path.join("acp"));
            assert!(scan.sessions.is_empty());
            assert_eq!(scan.keep_under, vec![looped]);
        }
    }

    #[test]
    fn is_session_store_matches_only_acp_layout() {
        assert!(is_session_store(Path::new(
            "/h/.cursor/acp-sessions/abc/store.db"
        )));
        assert!(!is_session_store(Path::new(
            "/h/.cursor/chats/hash/abc/store.db"
        )));
        assert!(!is_session_store(Path::new(
            "/h/.cursor/projects/p/agent-transcripts/abc/abc.jsonl"
        )));
    }

    #[test]
    fn parses_cursor_timestamps() {
        let ts = parse_cursor_timestamp("Thursday, Oct 1, 2026, 9:55 AM (UTC-7)").unwrap();
        assert_eq!(ts.to_rfc3339(), "2026-10-01T09:55:00-07:00");
        let ts = parse_cursor_timestamp("Wednesday, Oct 14, 2026, 11:05 PM (UTC+5:30)").unwrap();
        assert_eq!(ts.to_rfc3339(), "2026-10-14T23:05:00+05:30");
        let ts = parse_cursor_timestamp("Monday, Oct 12, 2026, 12:00 PM (UTC)").unwrap();
        assert_eq!(ts.to_rfc3339(), "2026-10-12T12:00:00+00:00");
        assert!(parse_cursor_timestamp("yesterday").is_none());
    }

    #[test]
    fn decode_fields_rejects_truncated_input() {
        assert!(decode_fields(&[0x0a, 0x05, 0x01]).is_none());
        assert!(decode_fields(&[0x08, 0x80]).is_none());
        assert!(decode_fields(&[]).unwrap().is_empty());
    }
}
