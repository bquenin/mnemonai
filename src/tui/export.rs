//! Conversation export functionality.
//!
//! This module provides functions to export conversations in different formats:
//! - Ledger format (formatted text with speaker names)
//! - Plain text (simple speaker: message format)
//! - Markdown (with headers for speakers)
//! - JSONL (raw format)
//!
//! Conversations can be exported to files or copied to the clipboard.
//! Export respects the current display settings for thinking blocks and tool calls.

use crate::claude::{ContentBlock, LogEntry, UserContent, UserMessage, extract_tool_result_text};
use crate::text_processing::process_command_message;
use crate::tool_format;
use crate::tui::viewer::read_log_entries;
use arboard::Clipboard;
use chrono::Local;
use std::fs;
use std::path::Path;

/// Export format options
#[derive(Clone, Copy, Debug)]
pub enum ExportFormat {
    Ledger,
    Plain,
    Markdown,
    Jsonl,
}

/// Formats whose content is generated from parsed log entries.
///
/// Raw JSONL is deliberately absent: a JSONL export copies the source
/// transcript file verbatim and must never go through entry-based generation.
#[derive(Clone, Copy, Debug)]
pub enum EntryFormat {
    Ledger,
    Plain,
    Markdown,
}

impl ExportFormat {
    /// All export formats in menu order. Single source of truth for the export
    /// menu: the menu count, labels, and index mapping all derive from this.
    pub const ALL: [ExportFormat; 4] = [
        ExportFormat::Ledger,
        ExportFormat::Plain,
        ExportFormat::Markdown,
        ExportFormat::Jsonl,
    ];

    /// Get format from menu option index (matches the order of `ALL`)
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    /// Human-readable menu label for this format
    pub fn label(&self) -> &'static str {
        match self {
            ExportFormat::Ledger => "Ledger (formatted)",
            ExportFormat::Plain => "Plain text",
            ExportFormat::Markdown => "Markdown",
            ExportFormat::Jsonl => "JSONL (raw)",
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Ledger | ExportFormat::Plain => "txt",
            ExportFormat::Markdown => "md",
            ExportFormat::Jsonl => "jsonl",
        }
    }

    /// The entry-based format for this export, or `None` for raw JSONL
    /// (which copies the transcript file instead of generating content)
    pub fn entry_format(&self) -> Option<EntryFormat> {
        match self {
            ExportFormat::Ledger => Some(EntryFormat::Ledger),
            ExportFormat::Plain => Some(EntryFormat::Plain),
            ExportFormat::Markdown => Some(EntryFormat::Markdown),
            ExportFormat::Jsonl => None,
        }
    }
}

/// Result of an export operation
pub struct ExportResult {
    pub message: String,
}

/// Options for export content generation
#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    pub show_tools: bool,
    pub show_thinking: bool,
    /// Label for assistant messages (e.g., "Claude" or "Cursor")
    pub assistant_label: String,
}

/// Write export content to a timestamped file in the current directory.
///
/// Shared by both the file-based export path and the entry-based export path
/// so the filename scheme and status messages stay identical.
pub fn save_to_file(content: &str, ext: &str) -> ExportResult {
    let timestamp = Local::now().format("%Y-%m-%d-%H%M%S");
    let filename = format!("conversation-{}.{}", timestamp, ext);

    match fs::write(&filename, content) {
        Ok(_) => ExportResult {
            message: format!("Exported to {}", filename),
        },
        Err(e) => ExportResult {
            message: format!("Failed to write: {}", e),
        },
    }
}

/// Copy text to the clipboard, returning a formatted status message on failure.
///
/// Shared by every clipboard action (export, path yank, session-id yank) so the
/// clipboard mechanics and error messages stay identical.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    match Clipboard::new() {
        Ok(mut clipboard) => clipboard
            .set_text(text)
            .map_err(|e| format!("Clipboard error: {}", e)),
        Err(e) => Err(format!("Clipboard unavailable: {}", e)),
    }
}

/// Export conversation to file
pub fn export_to_file(
    source_path: &Path,
    format: ExportFormat,
    options: ExportOptions,
) -> ExportResult {
    let content = match generate_content(source_path, format, options) {
        Ok(c) => c,
        Err(e) => {
            return ExportResult {
                message: format!("Failed to read: {}", e),
            };
        }
    };

    save_to_file(&content, format.extension())
}

/// Copy conversation to clipboard
pub fn export_to_clipboard(
    source_path: &Path,
    format: ExportFormat,
    options: ExportOptions,
) -> ExportResult {
    let content = match generate_content(source_path, format, options) {
        Ok(c) => c,
        Err(e) => {
            return ExportResult {
                message: format!("Failed to read: {}", e),
            };
        }
    };

    match copy_to_clipboard(&content) {
        Ok(()) => ExportResult {
            message: "Copied to clipboard".to_string(),
        },
        Err(message) => ExportResult { message },
    }
}

/// Generate content in the specified format from a file path
fn generate_content(
    source_path: &Path,
    format: ExportFormat,
    options: ExportOptions,
) -> std::io::Result<String> {
    match format.entry_format() {
        None => fs::read_to_string(source_path),
        Some(entry_format) => {
            let entries = read_log_entries(source_path)?;
            Ok(generate_content_from_entries(
                &entries,
                entry_format,
                options,
            ))
        }
    }
}

/// Generate content in the specified format from pre-parsed entries
pub fn generate_content_from_entries(
    entries: &[LogEntry],
    format: EntryFormat,
    options: ExportOptions,
) -> String {
    match format {
        EntryFormat::Plain => generate_plain_from_entries(entries, options),
        EntryFormat::Markdown => generate_markdown_from_entries(entries, options),
        EntryFormat::Ledger => generate_ledger_from_entries(entries, options),
    }
}

/// Generate plain text format from entries
fn generate_plain_from_entries(entries: &[LogEntry], options: ExportOptions) -> String {
    let mut output = String::new();

    for entry in entries {
        match entry {
            LogEntry::User { message, .. } => {
                if let Some(text) = extract_user_text(message) {
                    output.push_str(&format!("You: {}\n\n", text));
                }
                if options.show_tools
                    && let UserContent::Blocks(blocks) = &message.content
                {
                    for block in blocks {
                        if let ContentBlock::ToolResult { content, .. } = block {
                            let content_str = format_tool_result_for_export(content.as_ref());
                            output.push_str(&format!("Tool Result: {}\n\n", content_str));
                        }
                    }
                }
            }
            LogEntry::Assistant { message, .. } => {
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => {
                            output.push_str(&format!("{}: {}\n\n", options.assistant_label, text));
                        }
                        ContentBlock::ToolUse { name, input, .. } if options.show_tools => {
                            let formatted = format_tool_call_for_export(name, input);
                            output.push_str(&format!("Tool: {}\n\n", formatted));
                        }
                        ContentBlock::Thinking { thinking, .. } if options.show_thinking => {
                            output.push_str(&format!("Thinking: {}\n\n", thinking));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    output
}

/// Generate markdown format from entries
fn generate_markdown_from_entries(entries: &[LogEntry], options: ExportOptions) -> String {
    let mut output = String::new();

    for entry in entries {
        match entry {
            LogEntry::User { message, .. } => {
                if let Some(text) = extract_user_text(message) {
                    output.push_str(&format!("## You\n\n{}\n\n", text));
                }
                if options.show_tools
                    && let UserContent::Blocks(blocks) = &message.content
                {
                    for block in blocks {
                        if let ContentBlock::ToolResult { content, .. } = block {
                            let content_str = format_tool_result_for_export(content.as_ref());
                            let fenced = markdown_code_fence(&content_str);
                            output.push_str(&format!("### Tool Result\n\n{}\n\n", fenced));
                        }
                    }
                }
            }
            LogEntry::Assistant { message, .. } => {
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => {
                            output.push_str(&format!(
                                "## {}\n\n{}\n\n",
                                options.assistant_label, text
                            ));
                        }
                        ContentBlock::ToolUse { name, input, .. } if options.show_tools => {
                            let formatted = format_tool_call_for_export(name, input);
                            let fenced = markdown_code_fence(&formatted);
                            output.push_str(&format!("### Tool: {}\n\n{}\n\n", name, fenced));
                        }
                        ContentBlock::Thinking { thinking, .. } if options.show_thinking => {
                            output.push_str(&format!("### Thinking\n\n{}\n\n", thinking));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    output
}

/// Generate ledger-style format from entries
fn generate_ledger_from_entries(entries: &[LogEntry], options: ExportOptions) -> String {
    let mut output = String::new();
    const NAME_WIDTH: usize = 12;

    for entry in entries {
        match entry {
            LogEntry::User { message, .. } => {
                if let Some(text) = extract_user_text(message) {
                    append_ledger_block(&mut output, "You", &text, NAME_WIDTH);
                    output.push('\n');
                }
                if options.show_tools
                    && let UserContent::Blocks(blocks) = &message.content
                {
                    for block in blocks {
                        if let ContentBlock::ToolResult { content, .. } = block {
                            let content_str = format_tool_result_for_export(content.as_ref());
                            append_ledger_block(&mut output, "↳ Result", &content_str, NAME_WIDTH);
                            output.push('\n');
                        }
                    }
                }
            }
            LogEntry::Assistant { message, .. } => {
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => {
                            append_ledger_block(
                                &mut output,
                                &options.assistant_label,
                                text,
                                NAME_WIDTH,
                            );
                            output.push('\n');
                        }
                        ContentBlock::ToolUse { name, input, .. } if options.show_tools => {
                            let formatted = format_tool_call_for_export(name, input);
                            append_ledger_block(&mut output, "Tool", &formatted, NAME_WIDTH);
                            output.push('\n');
                        }
                        ContentBlock::Thinking { thinking, .. } if options.show_thinking => {
                            append_ledger_block(&mut output, "Thinking", thinking, NAME_WIDTH);
                            output.push('\n');
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    output
}

/// Append a ledger-formatted block to the output
fn append_ledger_block(output: &mut String, speaker: &str, text: &str, name_width: usize) {
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            output.push_str(&format!(
                "{:>width$} │ {}\n",
                speaker,
                line,
                width = name_width
            ));
        } else {
            output.push_str(&format!("{:>width$} │ {}\n", "", line, width = name_width));
        }
    }
}

/// Extract text from a user message, handling command messages
fn extract_user_text(message: &UserMessage) -> Option<String> {
    match &message.content {
        UserContent::String(s) => process_command_message(s),
        UserContent::Blocks(blocks) => {
            for block in blocks {
                if let ContentBlock::Text { text } = block
                    && let Some(processed) = process_command_message(text)
                {
                    return Some(processed);
                }
            }
            None
        }
    }
}

/// Wrap content in markdown code fence, handling nested backticks
fn markdown_code_fence(content: &str) -> String {
    // Find the longest run of backticks in content and use one more
    let max_backticks = content
        .split(|c| c != '`')
        .map(|s| s.len())
        .max()
        .unwrap_or(0);
    let fence_len = std::cmp::max(3, max_backticks + 1);
    let fence: String = std::iter::repeat_n('`', fence_len).collect();
    format!("{}\n{}\n{}", fence, content, fence)
}

/// Default width for export (no wrapping needed for markdown export)
const EXPORT_WIDTH: usize = usize::MAX;

/// Format a tool call for export
fn format_tool_call_for_export(name: &str, input: &serde_json::Value) -> String {
    // Use large width to avoid wrapping in export (full command on one line is better for copying)
    let formatted = tool_format::format_tool_call(name, input, EXPORT_WIDTH);
    match formatted.body {
        Some(body) => format!("{}\n{}", formatted.header, body),
        None => formatted.header,
    }
}

/// Format tool result content for export.
///
/// Text and text-block-array results share `extract_tool_result_text` with the
/// viewer; anything else falls back to pretty-printed JSON.
fn format_tool_result_for_export(content: Option<&serde_json::Value>) -> String {
    match extract_tool_result_text(content) {
        Some(text) => text,
        None => match content {
            Some(value) => {
                serde_json::to_string_pretty(value).unwrap_or_else(|_| "<error>".to_string())
            }
            None => "<no content>".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSONL export must return the transcript file verbatim, including lines
    /// that don't parse as log entries.
    #[test]
    fn generate_content_jsonl_returns_raw_file() {
        let dir =
            std::env::temp_dir().join(format!("mnemonai-export-jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("raw.jsonl");
        let raw = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}\n{\"not\":\"a log entry\"}\n";
        fs::write(&file, raw).unwrap();

        let content =
            generate_content(&file, ExportFormat::Jsonl, ExportOptions::default()).unwrap();

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(content, raw);
    }

    /// Raw JSONL is the only format without an entry-based generator, so it
    /// can never be routed through `generate_content_from_entries`.
    #[test]
    fn only_jsonl_has_no_entry_format() {
        assert!(matches!(
            ExportFormat::Ledger.entry_format(),
            Some(EntryFormat::Ledger)
        ));
        assert!(matches!(
            ExportFormat::Plain.entry_format(),
            Some(EntryFormat::Plain)
        ));
        assert!(matches!(
            ExportFormat::Markdown.entry_format(),
            Some(EntryFormat::Markdown)
        ));
        assert!(ExportFormat::Jsonl.entry_format().is_none());
    }

    fn user_string(text: &str) -> UserMessage {
        UserMessage {
            content: UserContent::String(text.to_string()),
        }
    }

    /// Export must skip `/clear` command messages, matching the viewer. Before
    /// consolidation the export copy displayed `/clear` as a normal command.
    #[test]
    fn extract_user_text_skips_clear_command() {
        assert_eq!(
            extract_user_text(&user_string("<command-name>/clear</command-name>")),
            None
        );
        assert_eq!(
            extract_user_text(&user_string("<command-name>/clear</command-name>")),
            process_command_message("<command-name>/clear</command-name>")
        );
    }

    /// Export must skip `<local-command-caveat>` blocks, matching the viewer.
    /// Before consolidation the export copy emitted the caveat text verbatim.
    #[test]
    fn extract_user_text_skips_local_command_caveat() {
        let caveat = "<local-command-caveat>Caveat: system generated.</local-command-caveat>";
        assert_eq!(extract_user_text(&user_string(caveat)), None);
        assert_eq!(
            extract_user_text(&user_string(caveat)),
            process_command_message(caveat)
        );
    }

    /// Export must unwrap Cursor-Agent `<user_query>`/`<timestamp>` tags,
    /// matching the viewer. Before consolidation the export copy kept the tags.
    #[test]
    fn extract_user_text_unwraps_cursor_tags() {
        let text = "<timestamp>Thursday, May 7, 2026, 10:12 PM (UTC-7)</timestamp>\n<user_query>\nperfect thanks\n</user_query>";
        assert_eq!(
            extract_user_text(&user_string(text)),
            Some("perfect thanks".to_string())
        );
        assert_eq!(
            extract_user_text(&user_string(text)),
            process_command_message(text)
        );
    }

    /// A non-command message still passes through unchanged.
    #[test]
    fn extract_user_text_preserves_normal_text() {
        assert_eq!(
            extract_user_text(&user_string("Hello world")),
            Some("Hello world".to_string())
        );
    }

    /// Text and text-block-array tool results must match `extract_tool_result_text`
    /// so export and viewer render identical content.
    #[test]
    fn tool_result_export_matches_extract_for_text() {
        let string_content = serde_json::json!("plain result");
        assert_eq!(
            format_tool_result_for_export(Some(&string_content)),
            extract_tool_result_text(Some(&string_content)).unwrap()
        );

        let array_content = serde_json::json!([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]);
        assert_eq!(
            format_tool_result_for_export(Some(&array_content)),
            "first\n\nsecond"
        );
        assert_eq!(
            format_tool_result_for_export(Some(&array_content)),
            extract_tool_result_text(Some(&array_content)).unwrap()
        );
    }

    /// Non-text JSON has no extractable text, so export falls back to
    /// pretty-printed JSON; missing content reports `<no content>`.
    #[test]
    fn tool_result_export_falls_back_to_pretty_json() {
        let object_content = serde_json::json!({"exit_code": 0});
        assert!(extract_tool_result_text(Some(&object_content)).is_none());
        assert_eq!(
            format_tool_result_for_export(Some(&object_content)),
            serde_json::to_string_pretty(&object_content).unwrap()
        );

        assert_eq!(format_tool_result_for_export(None), "<no content>");
    }
}
