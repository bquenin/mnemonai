//! Claude conversation history loading and parsing.
//!
//! This module provides functionality for:
//! - Loading conversations from Claude project directories
//! - Parsing JSONL conversation files
//! - Encoding/decoding project directory paths
//!
//! # Module Structure
//!
//! - `loader` - Loading conversations from directories
//! - `parser` - Parsing individual JSONL files
//! - `path` - Path encoding/decoding utilities

mod loader;
mod parser;
mod path;

use crate::error::{AppError, Result};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

// Re-export public API
pub use loader::{load_all_conversations, load_all_conversations_streaming};
pub use parser::process_conversation_file;
pub use path::{
    convert_path_to_project_dir_name, format_short_name_from_path, path_to_string,
    project_path_is_live, resolve_project_dir,
};

/// Identifies which AI tool provider a conversation originated from
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    Claude,
    Codex,
    Cursor,
    CursorAgent,
}

impl ProviderKind {
    /// Stable lowercase key used in the cache schema and the headless JSON
    /// output. Keep these strings stable; consumers depend on them.
    pub fn key(&self) -> &'static str {
        match self {
            ProviderKind::Claude => "claude",
            ProviderKind::Codex => "codex",
            ProviderKind::Cursor => "cursor",
            ProviderKind::CursorAgent => "cursor-agent",
        }
    }

    /// Human-readable provider name shown in the transcript view header.
    pub fn label(&self) -> &'static str {
        match self {
            ProviderKind::Claude => "Claude",
            ProviderKind::Codex => "Codex",
            ProviderKind::Cursor => "Cursor IDE",
            ProviderKind::CursorAgent => "Cursor Agent",
        }
    }

    /// Assistant name used to attribute assistant turns in exported
    /// transcripts.
    pub fn assistant_label(&self) -> &'static str {
        match self {
            ProviderKind::Claude => "Claude",
            ProviderKind::Codex => "Codex",
            ProviderKind::Cursor => "Cursor",
            ProviderKind::CursorAgent => "Cursor Agent",
        }
    }

    /// Bracketed badge text shown in list rows and transcript headers.
    /// Includes the trailing space.
    pub fn badge(&self) -> &'static str {
        match self {
            ProviderKind::Claude => "[Claude] ",
            ProviderKind::Codex => "[Codex] ",
            ProviderKind::Cursor => "[Cursor] ",
            ProviderKind::CursorAgent => "[Cursor CLI] ",
        }
    }

    /// Primary accent color (RGB) used for the provider badge and label.
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            ProviderKind::Claude => (218, 119, 86),
            ProviderKind::Codex => (78, 201, 176),
            ProviderKind::Cursor => (180, 130, 230),
            ProviderKind::CursorAgent => (94, 184, 255),
        }
    }

    /// Dimmed accent color (RGB) used for secondary provider styling.
    pub fn dim_color(&self) -> (u8, u8, u8) {
        match self {
            ProviderKind::Claude => (170, 93, 67),
            ProviderKind::Codex => (56, 150, 132),
            ProviderKind::Cursor => (140, 100, 180),
            ProviderKind::CursorAgent => (72, 140, 194),
        }
    }
}

/// Represents a JSONL parsing error with context for debugging
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParseError {
    pub line_number: usize,
    pub line_content: String,
    pub error_message: String,
    /// Lines before the error (up to 2)
    pub context_before: Vec<String>,
    /// Lines after the error (up to 2)
    pub context_after: Vec<String>,
}

#[derive(Clone)]
pub struct Conversation {
    pub path: PathBuf,
    /// Which provider this conversation belongs to
    pub provider: ProviderKind,
    /// Unique identifier (session UUID for Claude, composer/chat ID for Cursor)
    pub id: String,
    pub timestamp: DateTime<Local>,
    pub preview: String,
    pub full_text: String,
    pub project_name: Option<String>,
    pub project_path: Option<PathBuf>,
    /// The working directory extracted from the JSONL file (the actual cwd)
    pub cwd: Option<PathBuf>,
    /// Number of user and assistant messages in the conversation
    pub message_count: usize,
    /// Parse errors encountered while processing this conversation file
    pub parse_errors: Vec<ParseError>,
    /// Summary/title of the conversation (from type=summary JSONL entry)
    pub summary: Option<String>,
    /// Model name from assistant messages (e.g., "claude-opus-4-5-20251101")
    pub model: Option<String>,
    /// Total tokens used in the conversation (input + output + cache)
    pub total_tokens: u64,
    /// Conversation duration in minutes (from first to last message)
    pub duration_minutes: Option<u64>,
}

pub struct Project {
    pub name: String,         // directory name (encoded)
    pub display_name: String, // heuristic decoded path
    pub modified: SystemTime,
}

/// Message sent from background loader to TUI
pub enum LoaderMessage {
    /// A fatal error occurred (e.g., projects root doesn't exist)
    Fatal(AppError),
    /// A non-fatal error occurred (project-level, error already logged)
    ProjectError,
    /// A batch of loaded conversations from one project
    Batch(Vec<Conversation>),
    /// Loading completed
    Done,
}

/// Get the root Claude projects directory (~/.claude/projects)
pub fn get_claude_projects_root() -> Result<PathBuf> {
    claude_projects_root_from_home(home::home_dir())
}

fn claude_projects_root_from_home(home_dir: Option<PathBuf>) -> Result<PathBuf> {
    let home_dir = home_dir.ok_or_else(|| {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "home directory not found",
        ))
    })?;

    Ok(home_dir.join(".claude").join("projects"))
}

/// Get the Claude projects directory for the current working directory
pub fn get_claude_projects_dir(current_dir: &std::path::Path) -> Result<PathBuf> {
    let converted = convert_path_to_project_dir_name(current_dir);
    Ok(get_claude_projects_root()?.join(converted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_projects_root_uses_resolved_home_dir() {
        let home = PathBuf::from("home").join("user");

        let root = claude_projects_root_from_home(Some(home.clone())).unwrap();

        assert_eq!(root, home.join(".claude").join("projects"));
    }

    #[test]
    fn claude_projects_root_errors_when_home_dir_is_unavailable() {
        let error = claude_projects_root_from_home(None).unwrap_err();

        assert!(error.to_string().contains("home directory not found"));
    }

    #[test]
    fn provider_kind_metadata_is_pinned() {
        // Pins every user-visible provider string and color byte-for-byte.
        // (kind, key, label, assistant_label, badge, color, dim_color)
        let expected = [
            (
                ProviderKind::Claude,
                "claude",
                "Claude",
                "Claude",
                "[Claude] ",
                (218, 119, 86),
                (170, 93, 67),
            ),
            (
                ProviderKind::Codex,
                "codex",
                "Codex",
                "Codex",
                "[Codex] ",
                (78, 201, 176),
                (56, 150, 132),
            ),
            (
                ProviderKind::Cursor,
                "cursor",
                "Cursor IDE",
                "Cursor",
                "[Cursor] ",
                (180, 130, 230),
                (140, 100, 180),
            ),
            (
                ProviderKind::CursorAgent,
                "cursor-agent",
                "Cursor Agent",
                "Cursor Agent",
                "[Cursor CLI] ",
                (94, 184, 255),
                (72, 140, 194),
            ),
        ];

        for (kind, key, label, assistant_label, badge, color, dim_color) in expected {
            assert_eq!(kind.key(), key, "key mismatch for {kind:?}");
            assert_eq!(kind.label(), label, "label mismatch for {kind:?}");
            assert_eq!(
                kind.assistant_label(),
                assistant_label,
                "assistant_label mismatch for {kind:?}"
            );
            assert_eq!(kind.badge(), badge, "badge mismatch for {kind:?}");
            assert_eq!(kind.color(), color, "color mismatch for {kind:?}");
            assert_eq!(
                kind.dim_color(),
                dim_color,
                "dim_color mismatch for {kind:?}"
            );
        }
    }
}
