use crate::history::ProviderKind;
use clap::{Parser, Subcommand, ValueEnum};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

/// Log level for debug output filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DebugLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl FromStr for DebugLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "debug" => Ok(DebugLevel::Debug),
            "info" => Ok(DebugLevel::Info),
            "warn" | "warning" => Ok(DebugLevel::Warn),
            "error" => Ok(DebugLevel::Error),
            _ => Err(format!(
                "invalid log level '{}', expected: debug, info, warn, error",
                s
            )),
        }
    }
}

impl fmt::Display for DebugLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DebugLevel::Debug => write!(f, "debug"),
            DebugLevel::Info => write!(f, "info"),
            DebugLevel::Warn => write!(f, "warn"),
            DebugLevel::Error => write!(f, "error"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "mnemonai")]
#[command(version)]
#[command(about = "Universal AI coding conversation history browser")]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Show tool calls in the conversation output
    #[arg(long, short = 't', group = "tools_display")]
    pub show_tools: bool,

    /// Hide tool calls from the conversation output
    #[arg(long, group = "tools_display")]
    pub no_tools: bool,

    /// Show the conversation directory and exit
    #[arg(
        long,
        short = 'd',
        help = "Print the conversation directory path and exit"
    )]
    pub show_dir: bool,

    /// Show the last messages in the TUI preview
    #[arg(long, short = 'l', group = "preview_content")]
    pub last: bool,

    /// Show the first messages in the TUI preview
    #[arg(long, group = "preview_content")]
    pub first: bool,

    /// Display relative time (e.g. "10 minutes ago")
    #[arg(long, short = 'r', group = "time_display")]
    pub relative_time: bool,

    /// Display absolute timestamp
    #[arg(long, group = "time_display")]
    pub absolute_time: bool,

    /// Show thinking blocks in the conversation output
    #[arg(long, group = "thinking_display")]
    pub show_thinking: bool,

    /// Hide thinking blocks from the conversation output
    #[arg(long, group = "thinking_display")]
    pub hide_thinking: bool,

    /// Resume the selected conversation in its original tool
    #[arg(
        long,
        short = 'c',
        help = "Resume the selected conversation in its original tool"
    )]
    pub resume: bool,

    /// Print the selected conversation's file path and exit
    #[arg(long, short = 'p', help = "Print the selected conversation file path")]
    pub show_path: bool,

    /// Print the selected conversation's session ID and exit
    #[arg(long, short = 'i', help = "Print the selected conversation session ID")]
    pub show_id: bool,

    /// Output in plain text format without ledger formatting (for piping to other tools)
    #[arg(long, help = "Output plain text without ledger formatting")]
    pub plain: bool,

    /// Show debug output for conversation loading
    #[arg(
        long,
        value_name = "LEVEL",
        default_missing_value = "debug",
        num_args = 0..=1,
        help = "Print debug information (optionally filter by level: debug, info, warn, error)"
    )]
    pub debug: Option<DebugLevel>,

    /// Only show conversations from the current directory tree
    #[arg(
        long,
        short = 'L',
        help = "Only show conversations from the current directory tree"
    )]
    pub local: bool,

    /// Show all conversations, ignoring the current directory tree scope
    #[arg(
        long,
        conflicts_with = "local",
        help = "Show all conversations, ignoring the current directory tree scope"
    )]
    pub global: bool,

    /// Include conversations from deleted project directories
    #[arg(long, help = "Include conversations from deleted project directories")]
    pub show_deleted_projects: bool,

    /// Display output through a pager (less)
    #[arg(long, group = "pager_display")]
    pub pager: bool,

    /// Disable pager output
    #[arg(long, group = "pager_display")]
    pub no_pager: bool,

    /// Render a JSONL file in ledger format and exit (for debugging)
    #[arg(
        long,
        value_name = "FILE",
        help = "Render a JSONL file in ledger format and exit"
    )]
    pub render: Option<PathBuf>,

    /// Disable colored output (for --render)
    #[arg(long, help = "Disable colored output")]
    pub no_color: bool,

    /// Benchmark startup loading (headless) and exit
    #[arg(long, hide = true)]
    pub bench_startup: bool,

    /// Input JSONL file to view directly (skips conversation selection)
    #[arg(
        value_name = "FILE",
        help = "JSONL conversation file to view directly",
        conflicts_with_all = ["local", "global", "show_dir", "resume", "show_path", "show_id", "plain", "render"]
    )]
    pub input_file: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// List conversations without opening the TUI
    List(ListCommand),

    /// Show one conversation without opening the TUI
    Show(ShowCommand),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProviderFilter {
    Claude,
    Codex,
    Cursor,
    CursorAgent,
}

impl ProviderFilter {
    /// The provider kind this filter selects.
    pub fn kind(self) -> ProviderKind {
        match self {
            ProviderFilter::Claude => ProviderKind::Claude,
            ProviderFilter::Codex => ProviderKind::Codex,
            ProviderFilter::Cursor => ProviderKind::Cursor,
            ProviderFilter::CursorAgent => ProviderKind::CursorAgent,
        }
    }
}

#[derive(Parser, Debug)]
pub struct ListCommand {
    /// Output a JSON array (default)
    #[arg(long, group = "headless_list_output")]
    pub json: bool,

    /// Output one JSON object per line
    #[arg(long, group = "headless_list_output")]
    pub jsonl: bool,

    /// Only include conversations from this provider
    #[arg(long, value_enum)]
    pub provider: Option<ProviderFilter>,

    /// Only include conversations from the current directory tree
    #[arg(long)]
    pub local: bool,

    /// Only include conversations whose cwd or project path is at or under this path
    #[arg(long, value_name = "PATH", conflicts_with = "local")]
    pub cwd: Option<PathBuf>,

    /// Only include conversations from the last duration (for example: 7d, 24h, 2w)
    #[arg(long, value_name = "DURATION", conflicts_with = "after")]
    pub since: Option<String>,

    /// Only include conversations at or after this timestamp (RFC 3339 or YYYY-MM-DD)
    #[arg(long, value_name = "TIMESTAMP")]
    pub after: Option<String>,

    /// Only include conversations before this timestamp (RFC 3339 or YYYY-MM-DD)
    #[arg(long, value_name = "TIMESTAMP")]
    pub before: Option<String>,

    /// Include conversations from deleted project directories
    #[arg(long)]
    pub show_deleted_projects: bool,

    /// Limit the number of conversations returned
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Parser, Debug)]
pub struct ShowCommand {
    /// Conversation ID or source path
    pub target: String,

    /// Output structured JSON (default)
    #[arg(long)]
    pub json: bool,

    /// Only search conversations from this provider
    #[arg(long, value_enum)]
    pub provider: Option<ProviderFilter>,

    /// Only search conversations from the current directory tree
    #[arg(long)]
    pub local: bool,

    /// Include conversations from deleted project directories
    #[arg(long)]
    pub show_deleted_projects: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_list_subcommand() {
        let args = Args::try_parse_from([
            "mnemonai",
            "list",
            "--jsonl",
            "--provider",
            "cursor-agent",
            "--since",
            "7d",
            "--cwd",
            ".",
            "--limit",
            "10",
        ])
        .unwrap();

        match args.command {
            Some(Command::List(command)) => {
                assert!(command.jsonl);
                assert_eq!(command.provider, Some(ProviderFilter::CursorAgent));
                assert_eq!(command.since.as_deref(), Some("7d"));
                assert_eq!(command.cwd, Some(PathBuf::from(".")));
                assert_eq!(command.limit, Some(10));
            }
            other => panic!("expected list command, got {:?}", other),
        }
    }

    #[test]
    fn keeps_legacy_file_argument_mode() {
        let args = Args::try_parse_from(["mnemonai", "session.jsonl"]).unwrap();

        assert!(args.command.is_none());
        assert_eq!(args.input_file, Some(PathBuf::from("session.jsonl")));
    }

    #[test]
    fn parses_global_interactive_flag() {
        let args = Args::try_parse_from(["mnemonai", "--global"]).unwrap();

        assert!(args.global);
        assert!(!args.local);
    }

    #[test]
    fn global_conflicts_with_local() {
        let result = Args::try_parse_from(["mnemonai", "--global", "--local"]);

        assert!(result.is_err());
    }
}
