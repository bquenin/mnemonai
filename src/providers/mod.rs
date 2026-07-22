pub mod claude;
pub mod codex;
pub mod cursor;
pub mod cursor_agent;

use crate::claude::LogEntry;
use crate::error::Result;
use crate::history::{Conversation, LoaderMessage, ProviderKind};
use std::sync::mpsc::Receiver;

/// How a caller wants conversations loaded.
///
/// The two profiles that matter in practice:
/// - the TUI (and `--bench-startup`) needs each conversation's lowercased
///   `full_text` for in-app search, so it sets `include_full_text = true`;
/// - headless `list`/`show` never read `full_text`, so they set it `false` and
///   the loader neither decodes it from the cache nor keeps it in memory — the
///   cache row is still written complete, only the returned value is projected.
#[derive(Clone, Copy)]
pub struct LoadOptions {
    /// Preview from the last messages (`--last`) rather than the first.
    pub show_last: bool,
    pub debug: Option<crate::cli::DebugLevel>,
    /// Populate `Conversation::full_text` on the returned conversations.
    pub include_full_text: bool,
}

pub trait Provider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn name(&self) -> &str;

    /// Load conversations (synchronous, for single-project mode)
    fn load_conversations(&self, options: LoadOptions) -> Result<Vec<Conversation>>;

    /// Load conversations with streaming (for global mode)
    fn load_conversations_streaming(&self, options: LoadOptions) -> Receiver<LoaderMessage>;

    /// Read log entries for viewing/export (the core abstraction)
    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>>;

    /// Resume a conversation in the original tool
    fn resume(&self, conversation: &Conversation, default_args: &[String]) -> Result<()>;

    /// Delete a conversation
    fn delete(&self, conversation: &Conversation) -> Result<()>;
}
