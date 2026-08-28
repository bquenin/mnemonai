//! JSONL conversation file parsing.
//!
//! This module handles parsing Claude conversation JSONL files and extracting
//! conversation metadata like preview text, message counts, and working directory.

use super::{Conversation, ParseError, PreviewPair, ProviderKind};
use crate::claude::{LogEntry, TokenUsage, extract_text_from_assistant, extract_text_from_user};
use crate::cli::DebugLevel;
use crate::debug;
use crate::error::Result;
use chrono::{DateTime, Local};
use serde::Deserialize;
use serde::de::IgnoredAny;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::SystemTime;

/// Cap stored in [`ParseError`] line/context strings. A single corrupt JSONL
/// line can be many megabytes; without a cap that bloat lands in RAM and,
/// worse, in the persisted `parse_errors_json` cache column.
const PARSE_ERROR_MAX_BYTES: usize = 2048;

/// Truncate `s` to at most [`PARSE_ERROR_MAX_BYTES`] bytes on a UTF-8 char
/// boundary, appending a marker when truncation happens. Short strings are
/// returned unchanged.
fn truncate_for_error(s: &str) -> String {
    if s.len() <= PARSE_ERROR_MAX_BYTES {
        return s.to_string();
    }
    // Walk back to a char boundary at or below the cap.
    let mut end = PARSE_ERROR_MAX_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 16);
    out.push_str(&s[..end]);
    out.push_str("… [truncated]");
    out
}

// ---------------------------------------------------------------------------
// Scan-only deserialization types
//
// The startup scan reads every JSONL line of every conversation file, but only
// consumes a tiny slice of each line: the entry `type`, the timestamp, cwd,
// text blocks' `text`, the assistant model / usage / message-id, and the
// summary. The full `crate::claude::LogEntry` (used by the viewer/headless
// paths) is an internally-tagged enum, which forces serde to buffer every line
// into an owned value tree before it can even pick the variant, and it
// materializes payloads the scan never reads — `Progress.data`, `ToolUse.input`,
// `ToolResult.content` (can be megabytes), `Image.source` (base64 screenshots),
// `Thinking.thinking`, and the `System` fields.
//
// These slim types avoid that: a one-field probe classifies the line and lets
// us skip progress/system/snapshot/unknown entries entirely, and the
// user/assistant bodies use plain structs whose content blocks skip the heavy
// payloads (via `IgnoredAny` / presence markers) instead of allocating them.
//
// Correctness: the accept/reject contract must match `LogEntry` exactly so the
// parse-error *count* is identical. Where the slim path cannot decide a line as
// confidently as the full parse would (a malformed block, an odd field type),
// it falls back to a full `LogEntry` parse of that one line — rare enough to
// never touch the real-data hot path, but enough to guarantee parity.
// ---------------------------------------------------------------------------

/// Classifies a line by its `type` tag without materializing the body. Other
/// fields are skipped by serde (via `IgnoredAny`), so even a multi-megabyte
/// progress or tool-result line is dispatched cheaply.
#[derive(Deserialize)]
struct TypeProbe {
    #[serde(rename = "type")]
    kind: Option<String>,
}

/// Slim mirror of `LogEntry::User`. `timestamp` is required, matching the full
/// enum (a user line lacking it is a parse error in both paths).
#[derive(Deserialize)]
struct ScanUser {
    message: ScanUserMessage,
    timestamp: String,
    cwd: Option<String>,
}

#[derive(Deserialize)]
struct ScanUserMessage {
    content: ScanUserContent,
}

/// Mirrors `UserContent`: a bare string or an array of content blocks.
#[derive(Deserialize)]
#[serde(untagged)]
enum ScanUserContent {
    Str(String),
    Blocks(Vec<ScanBlock>),
}

/// Slim mirror of `LogEntry::Assistant`.
#[derive(Deserialize)]
struct ScanAssistant {
    message: ScanAssistantMessage,
    timestamp: String,
}

#[derive(Deserialize)]
struct ScanAssistantMessage {
    content: Vec<ScanBlock>,
    model: Option<String>,
    usage: Option<TokenUsage>,
    id: Option<String>,
}

/// Slim mirror of `LogEntry::Summary`.
#[derive(Deserialize)]
struct ScanSummary {
    summary: String,
}

/// Slim mirror of `LogEntry::System`. The scan ignores system entries, but the
/// full enum still validates their fields (`subtype` is a required string;
/// `level` / `durationMs` are typed options), so a malformed system line is a
/// parse error. Mirror that contract without materializing anything.
#[derive(Deserialize)]
struct ScanSystem {
    #[allow(dead_code)]
    subtype: StrPresent,
    #[allow(dead_code)]
    level: Option<StrPresent>,
    #[serde(rename = "durationMs")]
    #[allow(dead_code)]
    duration_ms: Option<u64>,
}

/// Slim mirror of `LogEntry::Progress`. `data` is required (any value, however
/// large); a progress line missing it is a parse error in the full enum.
/// Presence-check it without materializing the payload.
#[derive(Deserialize)]
struct ScanProgress {
    #[serde(deserialize_with = "de_present")]
    #[allow(dead_code)]
    data: bool,
}

/// A content block, parsed just far enough to (a) extract text-block text and
/// (b) validate the same required fields `crate::claude::ContentBlock` requires,
/// so accept/reject decisions match. Heavy payloads never materialize:
///
/// - `input` / `source` are `Value` in the full enum (accept any JSON value,
///   including `null`); here they are presence markers — `true` iff the field
///   is present (any value, incl. `null`) — so a null value is accepted exactly
///   as the full parse accepts it, without allocating the value.
/// - `thinking` is a required string in the full enum; here it is type-checked
///   as a string but the (potentially large) value is discarded.
/// - `content` (`Option<Value>` in the full enum) is omitted entirely: it is
///   optional and accepts any value type, so the full parse never rejects on
///   it — skipping it (serde ignores unknown fields) can never diverge, and a
///   multi-megabyte tool-result body is dropped without allocation.
///
/// Fields foreign to a block's actual variant are ignored by the full enum's
/// per-variant deserialization but type-checked here; when that (or a missing
/// required field, or an unknown/absent `type`) makes the slim decision
/// uncertain, the caller falls back to a full parse of the line.
#[derive(Deserialize)]
struct ScanBlock {
    #[serde(rename = "type")]
    kind: Option<String>,
    /// `Text.text` — the only value the scan consumes.
    text: Option<String>,
    /// `ToolUse.id` (small string).
    id: Option<String>,
    /// `ToolUse.name` (small string).
    name: Option<String>,
    /// `ToolUse.input` — presence only (any value, incl. `null`), not materialized.
    #[serde(default, deserialize_with = "de_present")]
    input: bool,
    /// `ToolResult.tool_use_id` (small string).
    tool_use_id: Option<String>,
    /// `ToolResult.is_error` (`Option<bool>` in the full enum). Never read, but
    /// kept and typed so a wrong-typed value (e.g. `"yes"`) fails to
    /// deserialize here exactly as it does in the full parse, routing the line
    /// to the fallback and preserving the parse-error verdict.
    #[allow(dead_code)]
    is_error: Option<bool>,
    /// `ToolResult.status` (`Option<String>` in the full enum). Never read; kept
    /// and typed for the same accept/reject-parity reason as `is_error`.
    #[allow(dead_code)]
    status: Option<String>,
    /// `Thinking.thinking` — type-checked as a string, value discarded.
    thinking: Option<StrPresent>,
    /// `Image.source` — presence only (any value, incl. `null`), not materialized.
    #[serde(default, deserialize_with = "de_present")]
    source: bool,
}

/// Deserializes any JSON value (object, array, string, number, bool, or `null`)
/// without allocating it, reporting only that the field was present. Used for
/// `Value`-typed fields the scan never reads.
fn de_present<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    IgnoredAny::deserialize(deserializer).map(|_| true)
}

/// Accepts a JSON string (matching `String`'s accept/reject contract) but
/// discards the content — used for required-but-unread string fields that may
/// be large (`thinking`).
struct StrPresent;

impl<'de> Deserialize<'de> for StrPresent {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = StrPresent;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E>(self, _: &str) -> std::result::Result<StrPresent, E> {
                Ok(StrPresent)
            }
            fn visit_string<E>(self, _: String) -> std::result::Result<StrPresent, E> {
                Ok(StrPresent)
            }
        }
        deserializer.deserialize_str(V)
    }
}

/// Outcome of scanning a user or assistant line's content blocks on the fast
/// path.
enum BlockScan {
    /// All blocks were valid per the full enum's contract; carries the joined
    /// text-block text (identical to `extract_text_from_blocks`).
    Text(String),
    /// A block was malformed in a way whose accept/reject verdict the slim path
    /// cannot reproduce exactly; the caller must fall back to a full parse.
    Fallback,
}

/// Join the text of valid text blocks, or signal a fallback if any block
/// violates the full `ContentBlock` contract (missing required field, or an
/// unknown/absent `type`). Called only after the block vec deserialized, so
/// present fields already type-check.
fn scan_blocks(blocks: &[ScanBlock]) -> BlockScan {
    // Mirror `extract_text_from_blocks`: single spaces between text-block texts.
    let mut texts: Vec<&str> = Vec::new();
    for block in blocks {
        match block.kind.as_deref() {
            Some("text") => match &block.text {
                Some(text) => texts.push(text.as_str()),
                None => return BlockScan::Fallback,
            },
            Some("tool_use") => {
                if block.id.is_none() || block.name.is_none() || !block.input {
                    return BlockScan::Fallback;
                }
            }
            Some("tool_result") => {
                if block.tool_use_id.is_none() {
                    return BlockScan::Fallback;
                }
            }
            Some("thinking") => {
                if block.thinking.is_none() {
                    return BlockScan::Fallback;
                }
            }
            Some("image") => {
                if !block.source {
                    return BlockScan::Fallback;
                }
            }
            // Unknown block type or a missing `type` tag — the full enum
            // rejects both; defer to it for the exact verdict.
            _ => return BlockScan::Fallback,
        }
    }
    BlockScan::Text(texts.join(" "))
}

/// Text extracted from one classified line, plus the metadata the scan needs.
struct LineData {
    text: String,
    timestamp: String,
    cwd: Option<String>,
    model: Option<String>,
    usage: Option<TokenUsage>,
    msg_id: Option<String>,
}

/// The classification of a single non-blank line for the scan loop.
enum ScanLine {
    /// A user line: extracted text + timestamp + cwd.
    User(LineData),
    /// An assistant line: extracted text + timestamp + model/usage/id.
    Assistant(LineData),
    /// A summary line's summary string.
    Summary(String),
    /// Progress / system / file-history-snapshot / unknown — ignored, like the
    /// full parse's `_ => {}` and `System { .. } => {}` arms.
    Ignore,
    /// The line failed to parse (invalid JSON, non-string/absent `type`, or a
    /// malformed user/assistant body) — a parse error, matching the full parse.
    ParseError(String),
}

/// Fall back to a full `LogEntry` parse of a single line and reduce it to the
/// same `ScanLine` the fast path would have produced — used only for lines the
/// slim path cannot decide on its own, guaranteeing exact parity with the
/// original per-line handling.
fn classify_via_full_parse(line: &str) -> ScanLine {
    match serde_json::from_str::<LogEntry>(line) {
        Ok(LogEntry::User {
            message,
            cwd,
            timestamp,
            ..
        }) => ScanLine::User(LineData {
            text: extract_text_from_user(&message),
            timestamp,
            cwd,
            model: None,
            usage: None,
            msg_id: None,
        }),
        Ok(LogEntry::Assistant { message, timestamp }) => ScanLine::Assistant(LineData {
            text: extract_text_from_assistant(&message),
            timestamp,
            cwd: None,
            model: message.model,
            usage: message.usage,
            msg_id: message.id,
        }),
        Ok(LogEntry::Summary { summary }) => ScanLine::Summary(summary),
        Ok(_) => ScanLine::Ignore,
        Err(e) => ScanLine::ParseError(e.to_string()),
    }
}

/// Classify a single non-blank line into a [`ScanLine`], preferring the slim
/// fast path and falling back to a full parse only when necessary.
fn classify_line(line: &str) -> ScanLine {
    let probe = match serde_json::from_str::<TypeProbe>(line) {
        Ok(probe) => probe,
        // Invalid JSON, or a `type` that is present but not a string: the full
        // enum rejects these too (invalid JSON / tag not a string). Match the
        // full parse's error message so `error_message` stays faithful.
        Err(_) => return classify_via_full_parse(line),
    };

    match probe.kind.as_deref() {
        Some("user") => match serde_json::from_str::<ScanUser>(line) {
            Ok(user) => {
                let text = match user.message.content {
                    ScanUserContent::Str(s) => s,
                    ScanUserContent::Blocks(blocks) => match scan_blocks(&blocks) {
                        BlockScan::Text(text) => text,
                        BlockScan::Fallback => return classify_via_full_parse(line),
                    },
                };
                ScanLine::User(LineData {
                    text,
                    timestamp: user.timestamp,
                    cwd: user.cwd,
                    model: None,
                    usage: None,
                    msg_id: None,
                })
            }
            Err(_) => classify_via_full_parse(line),
        },
        Some("assistant") => match serde_json::from_str::<ScanAssistant>(line) {
            Ok(assistant) => match scan_blocks(&assistant.message.content) {
                BlockScan::Text(text) => ScanLine::Assistant(LineData {
                    text,
                    timestamp: assistant.timestamp,
                    cwd: None,
                    model: assistant.message.model,
                    usage: assistant.message.usage,
                    msg_id: assistant.message.id,
                }),
                BlockScan::Fallback => classify_via_full_parse(line),
            },
            Err(_) => classify_via_full_parse(line),
        },
        Some("summary") => match serde_json::from_str::<ScanSummary>(line) {
            Ok(summary) => ScanLine::Summary(summary.summary),
            Err(_) => classify_via_full_parse(line),
        },
        // `system` and `progress` are ignored by the scan, but the full enum
        // still validates their bodies (`System.subtype` required and typed,
        // `Progress.data` required) — a malformed line is a parse error. Check
        // the same contract slimly and defer to the full parse on failure so
        // the recorded error is identical.
        Some("system") => match serde_json::from_str::<ScanSystem>(line) {
            Ok(_) => ScanLine::Ignore,
            Err(_) => classify_via_full_parse(line),
        },
        Some("progress") => match serde_json::from_str::<ScanProgress>(line) {
            Ok(_) => ScanLine::Ignore,
            Err(_) => classify_via_full_parse(line),
        },
        // `file-history-snapshot` (unit variant — any body accepted) and every
        // type the full enum maps to `Unknown` (`#[serde(other)]`) are ignored,
        // same as the original loop's `_ => {}` arm. A *missing* type tag
        // (`kind == None`) is a parse error in the full enum (missing field
        // `type`); defer so it's recorded identically.
        Some(_) => ScanLine::Ignore,
        None => classify_via_full_parse(line),
    }
}

/// Process a single conversation file and extract all necessary information.
///
/// Returns the conversation (with `preview` already selected for `show_last`)
/// alongside both preview strings, so a caller persisting to the cache can store
/// them and later serve either preview mode without re-parsing.
pub fn process_conversation_file(
    path: PathBuf,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<Option<(Conversation, PreviewPair)>> {
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    process_conversation_reader(path, reader, show_last, modified, debug_level)
}

/// Process a conversation from any BufRead source (for testability)
pub(crate) fn process_conversation_reader<R: BufRead>(
    path: PathBuf,
    reader: R,
    show_last: bool,
    modified: Option<SystemTime>,
    debug_level: Option<DebugLevel>,
) -> Result<Option<(Conversation, PreviewPair)>> {
    let filename = path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");

    // Full-text parts (summary excluded — prepended later), retained to build
    // `full_text` in one pass at the end.
    let mut all_parts: Vec<String> = Vec::new();
    // Preview uses only the first 3 (show_last = false) or last 3 (true)
    // preview messages, so we retain just those instead of every message.
    let mut preview_head: Vec<String> = Vec::new();
    let mut preview_tail: VecDeque<String> = VecDeque::with_capacity(4);
    let mut seen_real_user_message = false;
    let mut skip_next_assistant = false;
    let mut extracted_cwd: Option<PathBuf> = None;
    let mut message_count: usize = 0;
    let mut parse_errors: Vec<ParseError> = Vec::new();
    let mut extracted_summary: Option<String> = None;
    let mut extracted_model: Option<String> = None;
    // Track token usage per message ID to avoid double-counting streaming entries
    let mut token_usage_by_msg: HashMap<String, TokenUsage> = HashMap::new();
    let mut anonymous_token_count: u64 = 0;
    // Track first and last message timestamps for conversation duration. We
    // parse the first successful timestamp once, then only remember the *raw*
    // string of the most recent user/assistant line and parse it once at EOF —
    // the original code re-parsed every line.
    let mut first_timestamp: Option<chrono::DateTime<chrono::FixedOffset>> = None;
    let mut last_raw_timestamp = String::new();
    let mut have_last_raw = false;

    // Clear-only detection, computed inline instead of re-scanning a retained
    // `Vec<String>` of every user message at EOF. Mirrors
    // `is_clear_only_conversation`: a conversation is clear-only iff every
    // non-blank user message is a caveat / `/clear` / stdout marker and all
    // three marker kinds were seen.
    let mut clear_saw_caveat = false;
    let mut clear_saw_command = false;
    let mut clear_saw_stdout = false;
    let mut clear_saw_substantive = false;

    // Stream-parse lines with a ring buffer for error context (avoids loading entire file)
    // Ring buffer holds up to 2 previous lines for before-context
    let mut ring_buf: VecDeque<String> = VecDeque::with_capacity(3);
    // Pending errors that need after-context lines filled in
    let mut pending_errors: Vec<(ParseError, usize)> = Vec::new();

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };

        // Fill after-context for any pending errors (including blank lines)
        for (err, needed) in pending_errors.iter_mut() {
            if *needed > 0 {
                err.context_after.push(truncate_for_error(&line));
                *needed -= 1;
            }
        }
        // Drain completed pending errors into parse_errors, keep the rest
        let mut i = 0;
        while i < pending_errors.len() {
            if pending_errors[i].1 == 0 {
                let (error, _) = pending_errors.swap_remove(i);
                parse_errors.push(error);
            } else {
                i += 1;
            }
        }

        if !line.trim().is_empty() {
            match classify_line(&line) {
                ScanLine::User(data) => {
                    // Remember the raw timestamp; parse deferred to EOF.
                    last_raw_timestamp.clone_from(&data.timestamp);
                    have_last_raw = true;
                    if first_timestamp.is_none()
                        && let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&data.timestamp)
                    {
                        first_timestamp = Some(ts);
                    }

                    // Extract cwd from the first user message that has it
                    if extracted_cwd.is_none()
                        && let Some(cwd_str) = data.cwd
                    {
                        extracted_cwd = Some(PathBuf::from(cwd_str));
                    }

                    let text = data.text;
                    if !text.is_empty() {
                        // Inline clear-only tracking (replaces retaining every
                        // user message for an EOF re-scan).
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            let is_caveat = trimmed.starts_with(
                                "Caveat: The messages below were generated by the user while running local commands.",
                            );
                            let has_command_tag =
                                trimmed.contains("<command-name>/clear</command-name>");
                            let has_stdout_tag = trimmed.contains("<local-command-stdout>");
                            if is_caveat {
                                clear_saw_caveat = true;
                            }
                            if has_command_tag {
                                clear_saw_command = true;
                            }
                            if has_stdout_tag {
                                clear_saw_stdout = true;
                            }
                            if !(is_caveat || has_command_tag || has_stdout_tag) {
                                clear_saw_substantive = true;
                            }
                        }

                        if !is_clear_metadata_message(&text) {
                            // Check if this is a warmup message (first user message is "Warmup")
                            let is_warmup = !seen_real_user_message && text.trim() == "Warmup";
                            if is_warmup {
                                all_parts.push(text);
                                skip_next_assistant = true;
                            } else {
                                message_count += 1;
                                push_preview(&mut preview_head, &mut preview_tail, &text);
                                all_parts.push(text);
                                seen_real_user_message = true;
                            }
                        }
                    }
                }
                ScanLine::Assistant(data) => {
                    last_raw_timestamp.clone_from(&data.timestamp);
                    have_last_raw = true;
                    if first_timestamp.is_none()
                        && let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&data.timestamp)
                    {
                        first_timestamp = Some(ts);
                    }

                    // Extract model name from first assistant message that has it
                    if extracted_model.is_none()
                        && let Some(model) = data.model
                    {
                        extracted_model = Some(model);
                    }

                    // Track token usage by message ID to avoid double-counting.
                    // Multiple JSONL entries can exist for the same message
                    // (streaming); the entry API allocates the key once.
                    if let Some(usage) = data.usage {
                        if let Some(msg_id) = data.msg_id {
                            // Store/update usage for this message ID (last one wins)
                            token_usage_by_msg
                                .entry(msg_id)
                                .and_modify(|slot| *slot = usage.clone())
                                .or_insert(usage);
                        } else {
                            // No message ID - accumulate directly (legacy format)
                            anonymous_token_count += usage.input_tokens
                                + usage.output_tokens
                                + usage.cache_creation_input_tokens
                                + usage.cache_read_input_tokens;
                        }
                    }

                    let text = data.text;
                    if !text.is_empty() {
                        // Skip this assistant message if it follows a warmup user message
                        if skip_next_assistant {
                            skip_next_assistant = false;
                            all_parts.push(text);
                        } else if seen_real_user_message {
                            // Only add assistant messages to preview after we've seen a real user message
                            message_count += 1;
                            push_preview(&mut preview_head, &mut preview_tail, &text);
                            all_parts.push(text);
                        } else {
                            all_parts.push(text);
                        }
                    }
                }
                ScanLine::Summary(summary) => {
                    // Extract summary from the first summary entry
                    if extracted_summary.is_none() {
                        extracted_summary = Some(summary);
                    }
                }
                ScanLine::Ignore => {}
                ScanLine::ParseError(error_message) => {
                    // Capture parse error with ring buffer for before-context.
                    // Line and context are truncated so a corrupt multi-megabyte
                    // line can't bloat RAM or the persisted parse_errors cache.
                    let context_before: Vec<String> = ring_buf.iter().cloned().collect();

                    debug::warn(
                        debug_level,
                        &format!(
                            "Parse error in {} at line {}: {}",
                            filename,
                            line_idx + 1,
                            error_message
                        ),
                    );

                    let error = ParseError {
                        line_number: line_idx + 1, // 1-indexed for display
                        line_content: truncate_for_error(&line),
                        error_message,
                        context_before,
                        context_after: Vec::new(),
                    };

                    // Queue for after-context collection (up to 2 lines)
                    pending_errors.push((error, 2));
                }
            }
        }

        // Single ring buffer update point — all paths reach here. Context lines
        // are truncated at capture time to bound retained memory; short lines
        // (the common case) are moved, not copied.
        ring_buf.push_back(if line.len() <= PARSE_ERROR_MAX_BYTES {
            line
        } else {
            truncate_for_error(&line)
        });
        if ring_buf.len() > 2 {
            ring_buf.pop_front();
        }
    }

    // Flush any remaining pending errors
    for (error, _) in pending_errors {
        parse_errors.push(error);
    }

    // Check if this is a clear-only conversation or if preview is empty after filtering
    let is_clear_only =
        clear_saw_caveat && clear_saw_command && clear_saw_stdout && !clear_saw_substantive;
    if is_clear_only {
        debug::debug(
            debug_level,
            &format!("Filtered {}: clear-only conversation", filename),
        );
        return Ok(None);
    }

    let preview_is_empty = if show_last {
        preview_tail.is_empty()
    } else {
        preview_head.is_empty()
    };
    if all_parts.is_empty() || preview_is_empty {
        debug::debug(
            debug_level,
            &format!(
                "Filtered {}: empty conversation (all_parts={}, preview_parts={})",
                filename,
                all_parts.len(),
                if show_last {
                    preview_tail.len()
                } else {
                    preview_head.len()
                }
            ),
        );
        return Ok(None);
    }

    // Use file modification time, falling back to current time if unavailable
    let timestamp = modified
        .map(DateTime::<Local>::from)
        .unwrap_or_else(Local::now);

    // Create both previews in one pass so the cache can serve either mode. The
    // last-messages preview takes the last 3 in reverse order (newest first);
    // the first-messages preview takes the first 3 in order. This reproduces
    // `preview_parts.iter().rev().take(3)` / `.iter().take(3)` over the full
    // list, then a whole-string whitespace normalization.
    let previews = PreviewPair {
        first: normalize_whitespace(&preview_head.join(" ... ")),
        last: normalize_whitespace(
            &preview_tail
                .iter()
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ... "),
        ),
    };
    let preview = previews.select(show_last);

    // Build full_text in a single pass: normalized summary first (if any), then
    // each normalized part, joined by single spaces. This is byte-identical to
    // the previous `all_parts.join(" ")` + `format!("{} {}", summary, _)` +
    // whole-string `normalize_whitespace`, because normalization collapses all
    // whitespace runs and empty (whitespace-only) parts contribute nothing.
    let full_text = build_full_text(extracted_summary.as_deref(), &all_parts);

    // Sum token usage from deduplicated messages (all token types)
    let total_tokens: u64 = token_usage_by_msg
        .values()
        .map(|u| {
            u.input_tokens
                + u.output_tokens
                + u.cache_creation_input_tokens
                + u.cache_read_input_tokens
        })
        .sum::<u64>()
        + anonymous_token_count;

    // Resolve the last timestamp now: parse the last raw string once. If it
    // fails to parse (a malformed timestamp on the final user/assistant line —
    // real Claude data never emits this), fall back to the last timestamp we
    // know parsed, which is `first_timestamp`.
    let last_timestamp = if have_last_raw {
        chrono::DateTime::parse_from_rfc3339(&last_raw_timestamp)
            .ok()
            .or(first_timestamp)
    } else {
        None
    };

    // Calculate conversation duration in minutes
    let duration_minutes = match (first_timestamp, last_timestamp) {
        (Some(first), Some(last)) => {
            let duration = last.signed_duration_since(first);
            let minutes = duration.num_minutes();
            if minutes > 0 {
                Some(minutes as u64)
            } else {
                None
            }
        }
        _ => None,
    };

    // Extract session ID from filename stem (e.g., "abc123.jsonl" -> "abc123")
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(Some((
        Conversation {
            path,
            provider: ProviderKind::Claude,
            id,
            timestamp,
            preview,
            full_text,
            project_name: None,
            project_path: None,
            cwd: extracted_cwd,
            message_count,
            parse_errors,
            summary: extracted_summary,
            model: extracted_model,
            total_tokens,
            duration_minutes,
        },
        previews,
    )))
}

/// Record a preview message, retaining only what the preview can ever show:
/// the first 3 (for `show_last = false`) and the last 3 (for `show_last = true`).
fn push_preview(head: &mut Vec<String>, tail: &mut VecDeque<String>, text: &str) {
    if head.len() < 3 {
        head.push(text.to_string());
    }
    tail.push_back(text.to_string());
    if tail.len() > 3 {
        tail.pop_front();
    }
}

/// Append the whitespace-normalized tokens of `s` to `out`, separating from
/// existing content with a single space. Replicates `split_whitespace` exactly.
fn append_normalized(out: &mut String, s: &str) {
    for token in s.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(token);
    }
}

/// Build the search `full_text` in a single pre-sized pass: the summary (if
/// any) first, then every part, each whitespace-normalized and joined by single
/// spaces. Byte-identical to `normalize_whitespace(format!("{summary} {joined}"))`.
fn build_full_text(summary: Option<&str>, parts: &[String]) -> String {
    let capacity =
        summary.map_or(0, |s| s.len() + 1) + parts.iter().map(|p| p.len() + 1).sum::<usize>();
    let mut out = String::with_capacity(capacity);
    if let Some(summary) = summary {
        append_normalized(&mut out, summary);
    }
    for part in parts {
        append_normalized(&mut out, part);
    }
    out
}

/// Detects metadata emitted by the /clear command wrapper messages
pub(crate) fn is_clear_metadata_message(message: &str) -> bool {
    let trimmed = message.trim();

    trimmed.is_empty()
        || trimmed.starts_with(
            "Caveat: The messages below were generated by the user while running local commands.",
        )
        || trimmed.contains("<local-command-caveat>")
        || trimmed.contains("<command-name>/clear</command-name>")
        || trimmed.contains("<command-message>clear</command-message>")
        || trimmed.contains("<local-command-stdout>")
        || trimmed.contains("<command-args>")
}

/// Normalize whitespace in a string
pub(crate) fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<&str>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to create a user message JSON line
    fn user_msg(text: &str, cwd: Option<&str>) -> String {
        let cwd_json = match cwd {
            Some(c) => format!(r#""cwd": "{}","#, c),
            None => String::new(),
        };
        format!(
            r#"{{"type": "user", "timestamp": "2024-01-01T00:00:00Z", {}  "message": {{"role": "user", "content": "{}"}}}}"#,
            cwd_json, text
        )
    }

    /// Helper to create an assistant message JSON line
    fn assistant_msg(text: &str) -> String {
        format!(
            r#"{{"type": "assistant", "timestamp": "2024-01-01T00:00:00Z", "message": {{"role": "assistant", "content": [{{"type": "text", "text": "{}"}}]}}}}"#,
            text
        )
    }

    /// Helper to create an assistant message with model and usage
    fn assistant_msg_with_usage(
        text: &str,
        model: &str,
        input: u64,
        output: u64,
        cache_creation: u64,
        cache_read: u64,
    ) -> String {
        format!(
            r#"{{"type": "assistant", "timestamp": "2024-01-01T00:00:00Z", "message": {{"role": "assistant", "model": "{}", "usage": {{"input_tokens": {}, "output_tokens": {}, "cache_creation_input_tokens": {}, "cache_read_input_tokens": {}}}, "content": [{{"type": "text", "text": "{}"}}]}}}}"#,
            model, input, output, cache_creation, cache_read, text
        )
    }

    /// Helper to parse JSONL content, discarding the preview pair for the many
    /// tests that only assert on the conversation itself.
    fn parse_jsonl(content: &str) -> Result<Option<Conversation>> {
        let reader = Cursor::new(content);
        Ok(process_conversation_reader(
            PathBuf::from("test.jsonl"),
            reader,
            false, // show_last
            None,  // modified
            None,  // debug_level
        )?
        .map(|(conversation, _)| conversation))
    }

    /// Helper returning both the conversation and its preview pair.
    fn parse_jsonl_with_previews(
        content: &str,
        show_last: bool,
    ) -> Option<(Conversation, PreviewPair)> {
        let reader = Cursor::new(content);
        process_conversation_reader(PathBuf::from("test.jsonl"), reader, show_last, None, None)
            .unwrap()
    }

    // === Warmup message filtering ===

    #[test]
    fn filters_warmup_messages_from_preview() {
        let content = [
            user_msg("Warmup", None),
            assistant_msg("Ready"),
            user_msg("Hello world", None),
            assistant_msg("Hi there"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();

        // Preview should NOT include the warmup exchange
        assert!(!conv.preview.contains("Warmup"));
        assert!(!conv.preview.contains("Ready"));
        assert!(conv.preview.contains("Hello world"));
        assert!(conv.preview.contains("Hi there"));

        // But full_text SHOULD include warmup content for searching
        assert!(conv.full_text.contains("Warmup"));
        assert!(conv.full_text.contains("Ready"));
    }

    #[test]
    fn warmup_only_conversation_excluded_from_preview_but_preserved() {
        // A conversation with only warmup should still be valid if it has content
        let content = [
            user_msg("Warmup", None),
            assistant_msg("Ready"),
            user_msg("Actual question", None),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(!conv.preview.contains("Warmup"));
        assert!(conv.preview.contains("Actual question"));
    }

    // === Clear command filtering ===

    #[test]
    fn filters_clear_only_conversations() {
        let content = [
            user_msg(
                "Caveat: The messages below were generated by the user while running local commands.",
                None,
            ),
            user_msg("<command-name>/clear</command-name>", None),
            user_msg("<local-command-stdout></local-command-stdout>", None),
        ]
        .join("\n");

        let result = parse_jsonl(&content).unwrap();
        assert!(
            result.is_none(),
            "Clear-only conversation should be filtered"
        );
    }

    #[test]
    fn preserves_clear_command_in_mixed_conversation() {
        let content = [
            user_msg("Hello", None),
            assistant_msg("Hi"),
            user_msg(
                "Caveat: The messages below were generated by the user while running local commands.",
                None,
            ),
            user_msg("<command-name>/clear</command-name>", None),
            user_msg("Another question", None),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        // The conversation should be preserved since it has real content
        assert!(conv.preview.contains("Hello"));
        assert!(conv.preview.contains("Another question"));
    }

    // === CWD extraction ===

    #[test]
    fn extracts_cwd_from_first_user_message() {
        let content = [
            user_msg("Hello", Some("/home/user/project")),
            assistant_msg("Hi"),
            user_msg("More", Some("/other/path")),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.cwd,
            Some(PathBuf::from("/home/user/project")),
            "Should extract cwd from first user message"
        );
    }

    #[test]
    fn handles_missing_cwd() {
        let content = [user_msg("Hello", None), assistant_msg("Hi")].join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(conv.cwd.is_none());
    }

    // === Empty conversation handling ===

    #[test]
    fn handles_empty_conversation() {
        let content = "";
        let result = parse_jsonl(content).unwrap();
        assert!(result.is_none(), "Empty conversation should return None");
    }

    #[test]
    fn handles_only_whitespace() {
        let content = "\n\n   \n\n";
        let result = parse_jsonl(content).unwrap();
        assert!(result.is_none());
    }

    // === Message counting ===

    #[test]
    fn counts_messages_correctly() {
        let content = [
            user_msg("First", None),
            assistant_msg("Response 1"),
            user_msg("Second", None),
            assistant_msg("Response 2"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(conv.message_count, 4, "Should count 4 messages");
    }

    #[test]
    fn excludes_warmup_from_message_count() {
        let content = [
            user_msg("Warmup", None),
            assistant_msg("Ready"),
            user_msg("Real question", None),
            assistant_msg("Real answer"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        // Warmup and Ready should not be counted
        assert_eq!(
            conv.message_count, 2,
            "Should count 2 messages (excluding warmup)"
        );
    }

    // === Parse error handling ===

    #[test]
    fn captures_parse_errors_with_context() {
        let content = [
            user_msg("Line 1", None),
            "invalid json here".to_string(),
            user_msg("Line 3", None),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(conv.parse_errors.len(), 1);

        let error = &conv.parse_errors[0];
        assert_eq!(error.line_number, 2);
        assert!(error.line_content.contains("invalid json"));
        assert!(!error.error_message.is_empty());
        // Context before should have line 1
        assert_eq!(error.context_before.len(), 1);
        // Context after should have line 3
        assert_eq!(error.context_after.len(), 1);
    }

    // === Preview order ===

    #[test]
    fn show_last_reverses_preview_order() {
        let content = [
            user_msg("First", None),
            assistant_msg("Response 1"),
            user_msg("Second", None),
            assistant_msg("Response 2"),
            user_msg("Third", None),
            assistant_msg("Response 3"),
        ]
        .join("\n");

        // Parse with show_last = false
        let (conv_first, previews_first) = parse_jsonl_with_previews(&content, false).unwrap();

        // Parse with show_last = true
        let (conv_last, previews_last) = parse_jsonl_with_previews(&content, true).unwrap();

        // show_last=false should start with "First"
        assert!(
            conv_first.preview.starts_with("First"),
            "Preview should start with First: {}",
            conv_first.preview
        );

        // show_last=true should start with the last message (Response 3)
        assert!(
            conv_last.preview.starts_with("Response 3"),
            "Preview should start with Response 3: {}",
            conv_last.preview
        );

        // The preview pair is identical regardless of which mode was requested:
        // both previews are always computed, only `Conversation::preview` differs.
        assert_eq!(previews_first, previews_last);
        assert_eq!(conv_first.preview, previews_first.first);
        assert_eq!(conv_last.preview, previews_first.last);
        assert_eq!(
            previews_first.first, "First ... Response 1 ... Second",
            "first-messages preview pins the first three messages"
        );
        assert_eq!(
            previews_first.last, "Response 3 ... Third ... Response 2",
            "last-messages preview pins the last three in reverse order"
        );
    }

    // === Helper function tests ===

    #[test]
    fn is_clear_metadata_message_detects_patterns() {
        assert!(is_clear_metadata_message(""));
        assert!(is_clear_metadata_message("   "));
        assert!(is_clear_metadata_message(
            "Caveat: The messages below were generated by the user while running local commands."
        ));
        assert!(is_clear_metadata_message(
            "<local-command-caveat>something</local-command-caveat>"
        ));
        assert!(is_clear_metadata_message(
            "<command-name>/clear</command-name>"
        ));
        assert!(is_clear_metadata_message(
            "<command-message>clear</command-message>"
        ));
        assert!(is_clear_metadata_message(
            "<local-command-stdout>output</local-command-stdout>"
        ));
        assert!(is_clear_metadata_message(
            "<command-args>foo</command-args>"
        ));

        // Should NOT match normal messages
        assert!(!is_clear_metadata_message("Hello world"));
        assert!(!is_clear_metadata_message("What is the meaning of life?"));
    }

    #[test]
    fn normalize_whitespace_collapses_runs() {
        assert_eq!(normalize_whitespace("hello  world"), "hello world");
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("a\n\n\nb"), "a b");
        assert_eq!(
            normalize_whitespace("\t\thello\t\tworld\t\t"),
            "hello world"
        );
        assert_eq!(normalize_whitespace(""), "");
    }

    #[test]
    fn clear_only_requires_all_three_markers() {
        // The clear-only decision is now computed inline while scanning, so we
        // exercise it through the parser instead of a standalone helper.
        let caveat =
            "Caveat: The messages below were generated by the user while running local commands.";
        let command = "<command-name>/clear</command-name>";
        let stdout = "<local-command-stdout></local-command-stdout>";

        // Just caveat: not clear-only (but also has no real content -> None).
        let only_caveat = parse_jsonl(&user_msg(caveat, None)).unwrap();
        assert!(only_caveat.is_none());

        // Caveat + command but no stdout: not clear-only, and no real content.
        let two = [user_msg(caveat, None), user_msg(command, None)].join("\n");
        assert!(parse_jsonl(&two).unwrap().is_none());

        // All three markers = clear-only -> filtered out entirely.
        let all_three = [
            user_msg(caveat, None),
            user_msg(command, None),
            user_msg(stdout, None),
        ]
        .join("\n");
        assert!(
            parse_jsonl(&all_three).unwrap().is_none(),
            "all three markers should be treated as a clear-only conversation"
        );

        // All three markers plus a substantive message: NOT clear-only, so the
        // conversation is preserved and the substantive message survives.
        let with_substantive = [
            user_msg(caveat, None),
            user_msg(command, None),
            user_msg(stdout, None),
            user_msg("Hello world", None),
        ]
        .join("\n");
        let conv = parse_jsonl(&with_substantive).unwrap().unwrap();
        assert!(conv.preview.contains("Hello world"));
    }

    // === Summary extraction ===

    #[test]
    fn extracts_summary_from_jsonl() {
        let content = [
            r#"{"type": "summary", "summary": "Test conversation summary", "leafUuid": "abc123"}"#
                .to_string(),
            user_msg("Hello", None),
            assistant_msg("Hi there"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.summary,
            Some("Test conversation summary".to_string()),
            "Should extract summary from summary entry"
        );
    }

    #[test]
    fn summary_included_in_full_text() {
        let content = [
            r#"{"type": "summary", "summary": "Important topic discussion", "leafUuid": "abc123"}"#
                .to_string(),
            user_msg("Hello", None),
            assistant_msg("Hi there"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(
            conv.full_text.contains("Important topic discussion"),
            "Summary should be included in full_text for searching"
        );
    }

    #[test]
    fn handles_conversation_without_summary() {
        let content = [user_msg("Hello", None), assistant_msg("Hi there")].join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(conv.summary.is_none(), "Should have no summary");
    }

    #[test]
    fn takes_first_summary_if_multiple() {
        let content = [
            r#"{"type": "summary", "summary": "First summary", "leafUuid": "abc"}"#.to_string(),
            user_msg("Hello", None),
            r#"{"type": "summary", "summary": "Second summary", "leafUuid": "def"}"#.to_string(),
            assistant_msg("Hi there"),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.summary,
            Some("First summary".to_string()),
            "Should keep first summary encountered"
        );
    }

    // === Model and token extraction ===

    #[test]
    fn extracts_model_from_assistant_message() {
        let content = [
            user_msg("Hello", None),
            assistant_msg_with_usage("Hi there", "claude-opus-4-5-20251101", 100, 50, 0, 0),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.model,
            Some("claude-opus-4-5-20251101".to_string()),
            "Should extract model from assistant message"
        );
    }

    #[test]
    fn accumulates_tokens_across_messages() {
        let content = [
            user_msg("Hello", None),
            assistant_msg_with_usage("Hi", "claude-opus-4-5-20251101", 100, 50, 10, 5),
            user_msg("How are you?", None),
            assistant_msg_with_usage("Good!", "claude-opus-4-5-20251101", 200, 100, 20, 10),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        // Total = (100+50+10+5) + (200+100+20+10) = 495 (all token types)
        assert_eq!(
            conv.total_tokens, 495,
            "Should accumulate all token types from all assistant messages"
        );
    }

    #[test]
    fn takes_first_model_if_multiple() {
        let content = [
            user_msg("Hello", None),
            assistant_msg_with_usage("Hi", "claude-opus-4-5-20251101", 100, 50, 0, 0),
            user_msg("Follow up", None),
            assistant_msg_with_usage("Response", "claude-sonnet-4-20250514", 200, 100, 0, 0),
        ]
        .join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.model,
            Some("claude-opus-4-5-20251101".to_string()),
            "Should keep first model encountered"
        );
    }

    #[test]
    fn handles_missing_model_and_usage() {
        let content = [user_msg("Hello", None), assistant_msg("Hi there")].join("\n");

        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(conv.model.is_none(), "Should have no model");
        assert_eq!(conv.total_tokens, 0, "Should have zero tokens");
    }

    // =======================================================================
    // Equivalence oracle
    //
    // A faithful reimplementation of the ORIGINAL scan algorithm, built on the
    // untouched `crate::claude::LogEntry` and the shared `extract_text_from_*`
    // helpers. It reproduces main's exact preview / full_text / timestamp /
    // token / clear-only logic (the `all_parts.join(" ")` + `format!` +
    // whole-string `normalize_whitespace` pipeline included) so the optimized
    // scan can be checked field-by-field against it.
    // =======================================================================

    struct OracleOutput {
        preview: String,
        full_text: String,
        message_count: usize,
        model: Option<String>,
        total_tokens: u64,
        duration_minutes: Option<u64>,
        summary: Option<String>,
        cwd: Option<PathBuf>,
        timestamp_present: bool,
        parse_errors: usize,
        is_none: bool,
    }

    fn oracle(content: &str, show_last: bool) -> OracleOutput {
        use crate::claude::{
            LogEntry, TokenUsage, extract_text_from_assistant, extract_text_from_user,
        };
        use std::collections::HashMap;

        let mut all_parts: Vec<String> = Vec::new();
        let mut preview_parts: Vec<String> = Vec::new();
        let mut user_messages: Vec<String> = Vec::new();
        let mut seen_real_user_message = false;
        let mut skip_next_assistant = false;
        let mut extracted_cwd: Option<PathBuf> = None;
        let mut message_count: usize = 0;
        let mut parse_errors: usize = 0;
        let mut extracted_summary: Option<String> = None;
        let mut extracted_model: Option<String> = None;
        let mut token_usage_by_msg: HashMap<String, TokenUsage> = HashMap::new();
        let mut anonymous_token_count: u64 = 0;
        let mut first_timestamp: Option<chrono::DateTime<chrono::FixedOffset>> = None;
        let mut last_timestamp: Option<chrono::DateTime<chrono::FixedOffset>> = None;

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<LogEntry>(line) {
                Ok(LogEntry::User {
                    message,
                    cwd,
                    timestamp,
                    ..
                }) => {
                    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&timestamp) {
                        if first_timestamp.is_none() {
                            first_timestamp = Some(ts);
                        }
                        last_timestamp = Some(ts);
                    }
                    if extracted_cwd.is_none()
                        && let Some(cwd_str) = cwd
                    {
                        extracted_cwd = Some(PathBuf::from(cwd_str));
                    }
                    let text = extract_text_from_user(&message);
                    if !text.is_empty() {
                        user_messages.push(text.clone());
                        if !is_clear_metadata_message(&text) {
                            all_parts.push(text.clone());
                            let is_warmup = !seen_real_user_message && text.trim() == "Warmup";
                            if is_warmup {
                                skip_next_assistant = true;
                            } else {
                                message_count += 1;
                                preview_parts.push(text);
                                seen_real_user_message = true;
                            }
                        }
                    }
                }
                Ok(LogEntry::Assistant { message, timestamp }) => {
                    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&timestamp) {
                        if first_timestamp.is_none() {
                            first_timestamp = Some(ts);
                        }
                        last_timestamp = Some(ts);
                    }
                    if extracted_model.is_none()
                        && let Some(model) = &message.model
                    {
                        extracted_model = Some(model.clone());
                    }
                    if let Some(usage) = &message.usage {
                        if let Some(msg_id) = &message.id {
                            token_usage_by_msg.insert(msg_id.clone(), usage.clone());
                        } else {
                            anonymous_token_count += usage.input_tokens
                                + usage.output_tokens
                                + usage.cache_creation_input_tokens
                                + usage.cache_read_input_tokens;
                        }
                    }
                    let text = extract_text_from_assistant(&message);
                    if !text.is_empty() {
                        all_parts.push(text.clone());
                        if skip_next_assistant {
                            skip_next_assistant = false;
                        } else if seen_real_user_message {
                            message_count += 1;
                            preview_parts.push(text);
                        }
                    }
                }
                Ok(LogEntry::Summary { summary }) => {
                    if extracted_summary.is_none() {
                        extracted_summary = Some(summary.clone());
                    }
                }
                Ok(_) => {}
                Err(_) => parse_errors += 1,
            }
        }

        // Original clear-only check.
        let is_clear_only = {
            if user_messages.is_empty() {
                false
            } else {
                let mut saw_caveat = false;
                let mut saw_command = false;
                let mut saw_stdout = false;
                let mut substantive = false;
                for msg in &user_messages {
                    let trimmed = msg.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let is_caveat = trimmed.starts_with(
                        "Caveat: The messages below were generated by the user while running local commands.",
                    );
                    let has_command_tag = trimmed.contains("<command-name>/clear</command-name>");
                    let has_stdout_tag = trimmed.contains("<local-command-stdout>");
                    if is_caveat {
                        saw_caveat = true;
                    }
                    if has_command_tag {
                        saw_command = true;
                    }
                    if has_stdout_tag {
                        saw_stdout = true;
                    }
                    if !(is_caveat || has_command_tag || has_stdout_tag) {
                        substantive = true;
                    }
                }
                saw_caveat && saw_command && saw_stdout && !substantive
            }
        };

        let is_none = is_clear_only || all_parts.is_empty() || preview_parts.is_empty();

        // Original preview + full_text pipeline.
        let preview = if show_last {
            preview_parts
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ... ")
        } else {
            preview_parts
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ... ")
        };
        let mut full_text = all_parts.join(" ");
        if let Some(ref summary) = extracted_summary {
            full_text = format!("{} {}", summary, full_text);
        }
        let preview = normalize_whitespace(&preview);
        let full_text = normalize_whitespace(&full_text);

        let total_tokens: u64 = token_usage_by_msg
            .values()
            .map(|u| {
                u.input_tokens
                    + u.output_tokens
                    + u.cache_creation_input_tokens
                    + u.cache_read_input_tokens
            })
            .sum::<u64>()
            + anonymous_token_count;

        let duration_minutes = match (first_timestamp, last_timestamp) {
            (Some(first), Some(last)) => {
                let minutes = last.signed_duration_since(first).num_minutes();
                if minutes > 0 {
                    Some(minutes as u64)
                } else {
                    None
                }
            }
            _ => None,
        };

        OracleOutput {
            preview,
            full_text,
            message_count,
            model: extracted_model,
            total_tokens,
            duration_minutes,
            summary: extracted_summary,
            cwd: extracted_cwd,
            timestamp_present: true,
            parse_errors,
            is_none,
        }
    }

    /// Assert every cached Conversation field the optimized scan produces
    /// matches the oracle (main's algorithm) for the same input.
    fn assert_matches_oracle(content: &str, show_last: bool) {
        let expected = oracle(content, show_last);
        let reader = Cursor::new(content);
        let actual = process_conversation_reader(
            PathBuf::from("test.jsonl"),
            reader,
            show_last,
            Some(SystemTime::UNIX_EPOCH),
            None,
        )
        .unwrap();

        if expected.is_none {
            assert!(
                actual.is_none(),
                "expected filtered-out (None) conversation, got Some"
            );
            return;
        }

        let (actual, previews) = actual.expect("expected a conversation, got None");
        assert_eq!(actual.preview, expected.preview, "preview mismatch");
        assert_eq!(
            previews.select(show_last),
            actual.preview,
            "preview pair must agree with the selected preview"
        );
        assert_eq!(actual.full_text, expected.full_text, "full_text mismatch");
        assert_eq!(
            actual.message_count, expected.message_count,
            "message_count mismatch"
        );
        assert_eq!(actual.model, expected.model, "model mismatch");
        assert_eq!(
            actual.total_tokens, expected.total_tokens,
            "total_tokens mismatch"
        );
        assert_eq!(
            actual.duration_minutes, expected.duration_minutes,
            "duration mismatch"
        );
        assert_eq!(actual.summary, expected.summary, "summary mismatch");
        assert_eq!(actual.cwd, expected.cwd, "cwd mismatch");
        assert_eq!(
            actual.parse_errors.len(),
            expected.parse_errors,
            "parse_errors count mismatch"
        );
        assert!(expected.timestamp_present);
    }

    /// A rich fixture exercising every entry/block kind the scan must handle.
    fn rich_fixture() -> String {
        [
            // summary entry
            r#"{"type": "summary", "summary": "Fix the parser bug", "leafUuid": "s1"}"#.to_string(),
            // plain user string message with cwd
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "cwd": "/home/u/proj", "message": {"role": "user", "content": "How do I fix this?"}}"#.to_string(),
            // assistant with text + thinking + tool_use, with model/usage/id
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:05:00Z", "message": {"role": "assistant", "id": "m1", "model": "claude-opus-4-5-20251101", "usage": {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5}, "content": [{"type": "thinking", "thinking": "let me think about the approach here"}, {"type": "text", "text": "You should edit the file"}, {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"path": "a.rs", "content": "fn main() {}"}}]}}"#.to_string(),
            // streaming duplicate assistant id (usage dedup: last wins)
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:05:30Z", "message": {"role": "assistant", "id": "m1", "model": "claude-opus-4-5-20251101", "usage": {"input_tokens": 100, "output_tokens": 80, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5}, "content": [{"type": "text", "text": "You should edit the file carefully"}]}}"#.to_string(),
            // user with blocks including tool_result (large-ish content) + text
            r#"{"type": "user", "timestamp": "2024-01-01T00:06:00Z", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file edited ok", "is_error": false}, {"type": "text", "text": "Thanks, now run the tests"}]}}"#.to_string(),
            // assistant with an image block + text
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:07:00Z", "message": {"role": "assistant", "id": "m2", "model": "claude-opus-4-5-20251101", "usage": {"input_tokens": 20, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "Here is a screenshot"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANS"}}]}}"#.to_string(),
            // progress entry (plain)
            r#"{"type": "progress", "data": {"foo": "bar", "nested": {"a": [1, 2, 3]}}}"#.to_string(),
            // agent_progress progress entry
            r#"{"type": "progress", "data": {"type": "agent_progress", "agentId": "a1", "message": {"type": "assistant", "message": {"content": [{"type": "text", "text": "subagent text"}]}}}}"#.to_string(),
            // system entry
            r#"{"type": "system", "subtype": "turn_duration", "level": "info", "durationMs": 1234}"#.to_string(),
            // file-history-snapshot entry
            r#"{"type": "file-history-snapshot", "messageId": "x", "snapshot": {}}"#.to_string(),
            // unknown future type -> skipped, not an error
            r#"{"type": "ai-title", "title": "Some title"}"#.to_string(),
            // invalid JSON -> parse error
            "this is not json".to_string(),
            // trailing user message
            r#"{"type": "user", "timestamp": "2024-01-01T00:10:00Z", "message": {"role": "user", "content": "final question"}}"#.to_string(),
        ]
        .join("\n")
    }

    #[test]
    fn rich_fixture_matches_oracle_show_first() {
        assert_matches_oracle(&rich_fixture(), false);
    }

    /// `system` and `progress` bodies are validated by the full enum even
    /// though the scan ignores them: a system line missing `subtype` (or with a
    /// wrong-typed field) and a progress line missing `data` are parse errors,
    /// and the recorded message must be the full parse's own.
    #[test]
    fn malformed_system_and_progress_lines_match_oracle() {
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "hi"}}"#,
            r#"{"type": "system"}"#,
            r#"{"type": "system", "subtype": "turn_duration", "level": 5}"#,
            r#"{"type": "progress"}"#,
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}}"#,
        ]
        .join("\n");
        assert_matches_oracle(&content, false);
        assert_matches_oracle(&content, true);

        let (conv, _) = process_conversation_reader(
            PathBuf::from("test.jsonl"),
            Cursor::new(content.as_str()),
            false,
            Some(SystemTime::UNIX_EPOCH),
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(conv.parse_errors.len(), 3);
        assert_eq!(conv.parse_errors[0].line_number, 2);
        assert_eq!(conv.parse_errors[1].line_number, 3);
        assert_eq!(conv.parse_errors[2].line_number, 4);
        // Each recorded message is exactly what the full LogEntry parse says.
        for error in &conv.parse_errors {
            let expected = serde_json::from_str::<crate::claude::LogEntry>(&error.line_content)
                .unwrap_err()
                .to_string();
            assert_eq!(error.error_message, expected);
        }
    }

    /// Well-formed system and progress lines stay ignored (no parse errors, no
    /// effect on counts) — guards against the slim validators over-rejecting.
    #[test]
    fn well_formed_system_and_progress_lines_are_ignored() {
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "hi"}}"#,
            r#"{"type": "system", "subtype": "turn_duration", "level": "info", "durationMs": 42}"#,
            r#"{"type": "system", "subtype": "other", "level": null}"#,
            r#"{"type": "progress", "data": {"huge": [1, 2, 3], "nested": {"deep": "payload"}}}"#,
            r#"{"type": "progress", "data": null}"#,
        ]
        .join("\n");
        assert_matches_oracle(&content, false);

        let (conv, _) = process_conversation_reader(
            PathBuf::from("test.jsonl"),
            Cursor::new(content.as_str()),
            false,
            Some(SystemTime::UNIX_EPOCH),
            None,
        )
        .unwrap()
        .unwrap();
        assert!(conv.parse_errors.is_empty());
        assert_eq!(conv.message_count, 1);
    }

    #[test]
    fn rich_fixture_matches_oracle_show_last() {
        assert_matches_oracle(&rich_fixture(), true);
    }

    #[test]
    fn rich_fixture_field_values_are_stable() {
        // Lock the concrete values the fixture produces so accidental drift in
        // either the scan or the oracle is caught.
        let (conv, _) = {
            let reader = Cursor::new(rich_fixture());
            process_conversation_reader(
                PathBuf::from("test.jsonl"),
                reader,
                false,
                Some(SystemTime::UNIX_EPOCH),
                None,
            )
            .unwrap()
            .unwrap()
        };

        assert_eq!(conv.summary, Some("Fix the parser bug".to_string()));
        assert_eq!(conv.cwd, Some(PathBuf::from("/home/u/proj")));
        assert_eq!(conv.model, Some("claude-opus-4-5-20251101".to_string()));
        // Tokens: m1 dedup -> last (100+80+10+5=195), m2 (20+10=30) = 225.
        assert_eq!(conv.total_tokens, 225);
        // Duration: first 00:00:00 -> last 00:10:00 = 10 minutes.
        assert_eq!(conv.duration_minutes, Some(10));
        // Exactly one parse error (the invalid-JSON line); unknown type skipped.
        assert_eq!(conv.parse_errors.len(), 1);
        assert_eq!(conv.parse_errors[0].line_content, "this is not json");
        // message_count: user "How do I fix this?" (1), assistant m1 text (2),
        // assistant m1-dup text (3), user tool_result+text (4), assistant image
        // text (5), trailing user (6).
        assert_eq!(conv.message_count, 6);
        // full_text is summary + every non-empty text part, normalized.
        assert!(
            conv.full_text
                .starts_with("Fix the parser bug How do I fix this?")
        );
        assert!(
            conv.full_text
                .contains("You should edit the file carefully")
        );
        assert!(conv.full_text.contains("Here is a screenshot"));
        assert!(conv.full_text.contains("final question"));
        // Thinking / tool_use / tool_result / image payloads never leak into text.
        assert!(!conv.full_text.contains("let me think about the approach"));
        assert!(!conv.full_text.contains("iVBORw0KGgo"));
        assert!(!conv.full_text.contains("subagent text"));
    }

    // === full_text byte-identity (item B) ===

    /// The original full_text pipeline, kept verbatim as an oracle.
    fn original_full_text(summary: Option<&str>, all_parts: &[String]) -> String {
        let mut full_text = all_parts.join(" ");
        if let Some(summary) = summary {
            full_text = format!("{} {}", summary, full_text);
        }
        normalize_whitespace(&full_text)
    }

    #[test]
    fn build_full_text_is_byte_identical_to_original() {
        let cases: Vec<(Option<&str>, Vec<&str>)> = vec![
            (None, vec!["hello", "world"]),
            (None, vec!["  leading", "trailing  ", "  both  "]),
            (None, vec!["multiple   internal    spaces"]),
            (None, vec!["line\nwith\nnewlines", "and\ttabs"]),
            (None, vec!["a", "   ", "b"]), // whitespace-only middle part
            (None, vec!["   ", "   "]),    // all whitespace parts
            (Some("A summary here"), vec!["body one", "body two"]),
            (Some("  spaced summary  "), vec!["  spaced body  "]),
            (Some("   "), vec!["real body"]), // whitespace-only summary
            (Some(""), vec!["real body"]),    // empty summary
            (Some("only summary"), vec![]),   // no parts
            (Some("summary\nwith\nlines"), vec!["a\n\n\nb", "c"]),
        ];

        for (summary, parts_slices) in cases {
            let parts: Vec<String> = parts_slices.iter().map(|s| s.to_string()).collect();
            let expected = original_full_text(summary, &parts);
            let actual = build_full_text(summary, &parts);
            assert_eq!(
                actual, expected,
                "build_full_text diverged for summary={:?} parts={:?}",
                summary, parts_slices
            );
        }
    }

    // === Parse-error parity (item A) ===

    /// Count parse errors the ORIGINAL full-`LogEntry` scan would record, so we
    /// can assert the optimized scan produces exactly the same count.
    fn oracle_parse_error_count(content: &str) -> usize {
        oracle(content, false).parse_errors
    }

    /// Parse-error count the optimized scan records. Callers use inputs that
    /// yield a conversation (`Some`); a filtered `None` result would be a test
    /// setup bug, so we surface it loudly.
    fn scan_parse_error_count(content: &str) -> usize {
        let reader = Cursor::new(content);
        process_conversation_reader(
            PathBuf::from("test.jsonl"),
            reader,
            false,
            Some(SystemTime::UNIX_EPOCH),
            None,
        )
        .unwrap()
        .expect("test input should yield a conversation")
        .0
        .parse_errors
        .len()
    }

    #[test]
    fn malformed_tool_use_missing_id_is_parse_error_like_main() {
        // A tool_use block missing `id`: the full enum rejects the line, so it
        // is a parse error. The slim path must fall back and agree.
        let bad = r#"{"type": "assistant", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "name": "Edit", "input": {}}]}}"#;
        let content = [user_msg("Question", None), bad.to_string()].join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn unknown_content_block_type_is_parse_error_like_main() {
        // An unknown content block `type` is rejected by the full enum (no
        // `#[serde(other)]` on ContentBlock).
        let bad = r#"{"type": "assistant", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "future_block", "data": 1}]}}"#;
        let content = [user_msg("Question", None), bad.to_string()].join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
    }

    #[test]
    fn user_message_missing_timestamp_is_parse_error_like_main() {
        // `timestamp` is required on user/assistant lines in the full enum.
        let bad = r#"{"type": "user", "message": {"role": "user", "content": "no timestamp"}}"#;
        let content = [user_msg("Real", None), bad.to_string()].join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
    }

    #[test]
    fn image_with_null_source_is_accepted_like_main() {
        // The full enum's `source: Value` accepts JSON null, so an image block
        // with `"source": null` is NOT a parse error. The slim path treats the
        // field's presence (incl. null) as valid too.
        let content = [
            user_msg("Look", None),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "an image"}, {"type": "image", "source": null}]}}"#.to_string(),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 0);
        assert_eq!(scan_parse_error_count(&content), 0);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn text_block_with_foreign_wrongtyped_field_matches_main() {
        // A text block carrying a foreign field of the "wrong" type for another
        // variant (`thinking` as a number). The full enum's Text variant
        // ignores it and accepts; the slim path's flat struct fails to
        // deserialize, falls back to the full parse, and agrees (accept).
        let content = [
            user_msg("Q", None),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "answer", "thinking": 123}]}}"#.to_string(),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 0);
        assert_eq!(scan_parse_error_count(&content), 0);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn toolresult_wrongtyped_is_error_matches_main() {
        // tool_result with `is_error` as a string: the full enum requires
        // `Option<bool>`, so it rejects -> parse error. The slim path type-checks
        // `is_error` identically and agrees.
        let content = [
            user_msg("Q", None),
            r#"{"type": "user", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": "nope"}]}}"#.to_string(),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
    }

    #[test]
    fn multiple_parse_errors_count_matches_main() {
        let content = [
            user_msg("Start", None),
            "garbage line one".to_string(),
            assistant_msg("ok"),
            "{ still not valid json".to_string(),
            user_msg("End", None),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 2);
        assert_eq!(scan_parse_error_count(&content), 2);
        assert_matches_oracle(&content, false);
    }

    // === Parse-error truncation (item D) ===

    #[test]
    fn parse_error_content_is_truncated() {
        let huge = "x".repeat(10_000);
        // Build an invalid-JSON line that is huge (leading '{' then junk so it
        // fails to parse but isn't blank).
        let bad_line = format!("{{{}", huge);
        let content = [user_msg("Hi", None), bad_line, user_msg("Bye", None)].join("\n");

        let (conv, _) = {
            let reader = Cursor::new(&content);
            process_conversation_reader(PathBuf::from("t.jsonl"), reader, false, None, None)
                .unwrap()
                .unwrap()
        };
        assert_eq!(conv.parse_errors.len(), 1);
        let err = &conv.parse_errors[0];
        assert!(
            err.line_content.len() <= PARSE_ERROR_MAX_BYTES + 32,
            "line_content should be truncated, got {} bytes",
            err.line_content.len()
        );
        assert!(err.line_content.ends_with("… [truncated]"));
    }

    #[test]
    fn parse_error_context_lines_are_truncated() {
        // A huge (valid) surrounding line becomes truncated context.
        let huge_text = "y".repeat(10_000);
        let content = [
            user_msg(&huge_text, None),
            "not json".to_string(),
            user_msg("after", None),
        ]
        .join("\n");
        let (conv, _) = {
            let reader = Cursor::new(&content);
            process_conversation_reader(PathBuf::from("t.jsonl"), reader, false, None, None)
                .unwrap()
                .unwrap()
        };
        assert_eq!(conv.parse_errors.len(), 1);
        let err = &conv.parse_errors[0];
        assert_eq!(err.context_before.len(), 1);
        assert!(
            err.context_before[0].len() <= PARSE_ERROR_MAX_BYTES + 32,
            "context_before should be truncated"
        );
        assert!(err.context_before[0].ends_with("… [truncated]"));
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        // A multi-byte char straddling the cap must not panic or split.
        let s = "é".repeat(2000); // 2 bytes each -> 4000 bytes
        let out = truncate_for_error(&s);
        assert!(out.ends_with("… [truncated]"));
        // The retained prefix must be valid UTF-8 (String guarantees it; the
        // point is no panic and no partial code unit).
        assert!(out.len() <= PARSE_ERROR_MAX_BYTES + 32);
    }

    #[test]
    fn short_line_is_not_truncated() {
        assert_eq!(truncate_for_error("small"), "small");
    }

    // === Timestamp handling (item C) ===

    #[test]
    fn duration_uses_first_and_last_timestamps() {
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "start"}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:03:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "mid"}]}}"#.to_string(),
            r#"{"type": "user", "timestamp": "2024-01-01T00:42:00Z", "message": {"role": "user", "content": "end"}}"#.to_string(),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(conv.duration_minutes, Some(42));
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn duration_with_valid_trailing_timestamp_matches_main() {
        // Common case: every timestamp valid, last line valid -> scan's
        // deferred EOF parse equals main's per-line last.
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "a"}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:15:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "b"}]}}"#.to_string(),
        ]
        .join("\n");
        assert_matches_oracle(&content, false);
        assert_eq!(
            parse_jsonl(&content).unwrap().unwrap().duration_minutes,
            Some(15)
        );
    }

    #[test]
    fn single_timestamp_yields_no_duration() {
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "only"}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "same time"}]}}"#.to_string(),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(conv.duration_minutes, None);
    }

    // === Token dedup across streaming entries (item E) ===

    #[test]
    fn streaming_duplicate_ids_dedup_last_wins() {
        let content = [
            user_msg("Q", None),
            // Same id "m1" three times; only the last usage should count.
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "id": "m1", "model": "claude-opus-4-5-20251101", "usage": {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "partial"}]}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:01Z", "message": {"role": "assistant", "id": "m1", "usage": {"input_tokens": 5, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "more"}]}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:02Z", "message": {"role": "assistant", "id": "m1", "usage": {"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "final"}]}}"#.to_string(),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(
            conv.total_tokens, 30,
            "only the last usage for id m1 counts"
        );
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn anonymous_usage_accumulates() {
        // Assistant messages without an `id` accumulate directly.
        let content = [
            user_msg("Q", None),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "usage": {"input_tokens": 3, "output_tokens": 4, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "one"}]}}"#.to_string(),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:02:00Z", "message": {"role": "assistant", "usage": {"input_tokens": 1, "output_tokens": 2, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "content": [{"type": "text", "text": "two"}]}}"#.to_string(),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert_eq!(conv.total_tokens, 10);
        assert_matches_oracle(&content, false);
    }

    // === Preview head/tail caps (item B) ===

    #[test]
    fn preview_caps_first_three_and_last_three() {
        // Six exchanges -> twelve preview messages; preview keeps only 3.
        let mut lines = Vec::new();
        for i in 0..6 {
            lines.push(user_msg(&format!("user{}", i), None));
            lines.push(assistant_msg(&format!("asst{}", i)));
        }
        let content = lines.join("\n");

        // A single parse computes both previews; pin each mode byte-for-byte.
        let (first, previews) = parse_jsonl_with_previews(&content, false).unwrap();
        assert_eq!(first.preview, "user0 ... asst0 ... user1");
        assert_eq!(previews.first, "user0 ... asst0 ... user1");
        assert_eq!(previews.last, "asst5 ... user5 ... asst4");
        assert_matches_oracle(&content, false);

        // show_last = true selects the last-messages preview from the same pair.
        let (last, previews_last) = parse_jsonl_with_previews(&content, true).unwrap();
        assert_eq!(last.preview, "asst5 ... user5 ... asst4");
        assert_eq!(previews_last, previews);
        assert_matches_oracle(&content, true);
    }

    #[test]
    fn preview_with_whitespace_only_assistant_part_matches_main() {
        // An assistant message whose text is spaces-only is non-empty (so it is
        // retained) but normalizes away inside the preview join.
        let content = [
            user_msg("first", None),
            r#"{"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "   "}]}}"#.to_string(),
            user_msg("third", None),
        ]
        .join("\n");
        assert_matches_oracle(&content, false);
        assert_matches_oracle(&content, true);
    }

    #[test]
    fn empty_and_summary_only_conversations_are_none() {
        // Summary with no user/assistant messages -> filtered (all_parts empty).
        let content = r#"{"type": "summary", "summary": "orphan", "leafUuid": "x"}"#.to_string();
        assert!(parse_jsonl(&content).unwrap().is_none());
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn leading_assistant_before_user_excluded_from_preview_matches_main() {
        // Assistant text before any real user message goes into full_text but
        // not the preview/message_count.
        let content = [
            assistant_msg("leading assistant"),
            user_msg("the question", None),
            assistant_msg("the answer"),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(!conv.preview.contains("leading assistant"));
        assert!(conv.full_text.contains("leading assistant"));
        assert_eq!(conv.message_count, 2);
        assert_matches_oracle(&content, false);
    }

    // === Probe fallback parity (item A) ===

    #[test]
    fn non_string_type_tag_is_parse_error_like_main() {
        // `"type": 123` — the probe can't read a string tag, so it falls back
        // to the full parse, which rejects it (tag must be a string).
        let content = [
            user_msg("Q", None),
            r#"{"type": 123, "message": {}}"#.to_string(),
            user_msg("A", None),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn valid_json_non_object_line_is_parse_error_like_main() {
        // A syntactically valid JSON value that isn't an object (an array).
        let content = [
            user_msg("Q", None),
            "[1, 2, 3]".to_string(),
            user_msg("A", None),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn missing_type_tag_is_parse_error_like_main() {
        // An object with no `type` field — the full enum rejects (missing field
        // `type`); the probe's `kind == None` routes to the fallback and agrees.
        let content = [
            user_msg("Q", None),
            r#"{"message": {"role": "user", "content": "orphan"}}"#.to_string(),
            user_msg("A", None),
        ]
        .join("\n");
        assert_eq!(oracle_parse_error_count(&content), 1);
        assert_eq!(scan_parse_error_count(&content), 1);
        assert_matches_oracle(&content, false);
    }

    #[test]
    fn user_content_string_and_blocks_both_extract_text() {
        // A user line whose content is a bare string, and another whose content
        // is a text-block array — both should contribute their text.
        let content = [
            r#"{"type": "user", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": "bare string"}}"#.to_string(),
            r#"{"type": "user", "timestamp": "2024-01-01T00:01:00Z", "message": {"role": "user", "content": [{"type": "text", "text": "block text"}]}}"#.to_string(),
            assistant_msg("reply"),
        ]
        .join("\n");
        let conv = parse_jsonl(&content).unwrap().unwrap();
        assert!(conv.full_text.contains("bare string"));
        assert!(conv.full_text.contains("block text"));
        assert_matches_oracle(&content, false);
    }
}
