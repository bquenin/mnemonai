use crate::claude::LogEntry;
use crate::error::Result;
use crate::history::ProviderKind;
use crate::pager;
use crate::tui::{RenderOptions, RenderedLine, ToolDisplayMode, provider_theme, render_entries};
use colored::{Colorize, CustomColor};
use crossterm::terminal;
use std::io::{self, Write};
use std::path::Path;

/// Configuration options for displaying conversations
#[derive(Debug, Clone, Default)]
pub struct DisplayOptions {
    /// Hide tool calls and results
    pub no_tools: bool,
    /// Show thinking/reasoning blocks
    pub show_thinking: bool,
    /// Use a pager for output (less/more)
    pub use_pager: bool,
    /// Disable colored output (also used for `--plain`, a no-color ledger)
    pub no_color: bool,
}

const NAME_WIDTH: usize = 12;
const SEPARATOR_WIDTH: usize = 3; // Display width of " │ "

/// Claude's terracotta theme, used for the file-based `--render` path where no
/// provider context is available.
const CLAUDE_COLOR: (u8, u8, u8) = (218, 119, 86);
const CLAUDE_DIM_COLOR: (u8, u8, u8) = (170, 93, 67);

/// Get the terminal width, defaulting to 80 if unavailable
fn get_terminal_width() -> usize {
    terminal::size().map(|(w, _)| w as usize).unwrap_or(80)
}

/// Content width available after the ledger name column and separator.
fn content_width() -> usize {
    get_terminal_width().saturating_sub(NAME_WIDTH + SEPARATOR_WIDTH)
}

/// Build render options for the viewer-based ledger renderer.
fn render_options(
    label: String,
    color: (u8, u8, u8),
    dim_color: (u8, u8, u8),
    options: &DisplayOptions,
) -> RenderOptions {
    RenderOptions {
        tool_display: if options.no_tools {
            ToolDisplayMode::Hidden
        } else {
            ToolDisplayMode::Full
        },
        show_thinking: options.show_thinking,
        show_timing: false, // Non-TUI render doesn't support timing toggle
        content_width: content_width(),
        assistant_label: label,
        assistant_color: color,
        assistant_dim_color: dim_color,
    }
}

/// Render pre-parsed entries into viewer ledger lines for a given provider.
///
/// Shared by [`display_conversation`] and its tests so the exact lines the
/// terminal receives are the ones under test.
fn render_for_display(
    entries: &[LogEntry],
    provider_kind: &ProviderKind,
    options: &DisplayOptions,
) -> Vec<RenderedLine> {
    let (label, color, dim_color) = provider_theme(provider_kind);
    let render_options = render_options(label, color, dim_color, options);
    render_entries(entries, &render_options)
}

/// Convert rendered viewer lines to colored terminal output, honoring the
/// pager, `--no-color`, and BrokenPipe (pager quit) behaviors.
fn write_rendered_lines(rendered_lines: &[RenderedLine], options: &DisplayOptions) -> Result<()> {
    // Spawn pager if requested
    let mut pager_child = if options.use_pager {
        pager::spawn_pager().ok()
    } else {
        None
    };

    // Get writer - either pager stdin or stdout
    let mut stdout_handle = io::stdout().lock();
    let writer: &mut dyn Write = if let Some(ref mut child) = pager_child {
        child.stdin.as_mut().unwrap()
    } else {
        &mut stdout_handle
    };

    // Convert RenderedLine spans to colored terminal output
    'outer: for line in rendered_lines {
        for (text, style) in &line.spans {
            // Apply styling only if colors are enabled
            let output: Box<dyn std::fmt::Display> = if options.no_color {
                Box::new(text.as_str())
            } else {
                let mut styled = text.as_str().normal();

                if let Some((r, g, b)) = style.fg {
                    styled = styled.custom_color(CustomColor { r, g, b });
                }
                if style.bold {
                    styled = styled.bold();
                }
                if style.dimmed {
                    styled = styled.dimmed();
                }
                if style.italic {
                    styled = styled.italic();
                }

                Box::new(styled)
            };

            // Stop if the output pipe is closed (e.g., pager quit)
            if write!(writer, "{}", output).is_err() {
                break 'outer;
            }
        }
        if writeln!(writer).is_err() {
            break;
        }
    }

    // Close stdin and wait for pager to finish
    drop(stdout_handle);
    if let Some(mut child) = pager_child {
        let _ = child.wait();
    }

    Ok(())
}

/// Display a selected conversation's entries in the ledger format.
///
/// Entries come from the owning provider's `read_entries`, so non-Claude and
/// SQLite-backed (Cursor) conversations render correctly and are labeled with
/// the right per-provider name/colors. Uses the same viewer-based renderer as
/// the `--render` path. `--plain` maps to `no_color` for a colorless ledger.
pub fn display_conversation(
    entries: &[LogEntry],
    provider_kind: &ProviderKind,
    options: &DisplayOptions,
) -> Result<()> {
    let rendered_lines = render_for_display(entries, provider_kind, options);
    write_rendered_lines(&rendered_lines, options)
}

/// Render a conversation JSONL file in TUI ledger format to the terminal.
///
/// Backs the `--render <file>` path: the file is parsed directly as Claude
/// `LogEntry` JSONL and labeled "Claude", with no provider context.
pub fn render_to_terminal(file_path: &Path, options: &DisplayOptions) -> Result<()> {
    use crate::tui::render_conversation;

    let render_options = render_options(
        "Claude".to_string(),
        CLAUDE_COLOR,
        CLAUDE_DIM_COLOR,
        options,
    );

    let rendered_lines = render_conversation(file_path, &render_options)?;
    write_rendered_lines(&rendered_lines, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<LogEntry> {
        // A minimal user/assistant exchange. Providers reconstruct entries into
        // this same Claude `LogEntry` shape, so only the provider *kind* (and
        // thus the label) should differ between transcripts.
        let jsonl = [
            r#"{"type":"user","timestamp":"2024-01-01T00:00:00Z","message":{"content":"hi there"}}"#,
            r#"{"type":"assistant","timestamp":"2024-01-01T00:00:01Z","message":{"content":[{"type":"text","text":"hello back"}]}}"#,
        ];
        jsonl
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// First-span labels (the ledger name column) with padding trimmed.
    fn labels(lines: &[RenderedLine]) -> Vec<String> {
        lines
            .iter()
            .filter_map(|l| l.spans.first())
            .map(|(t, _)| t.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn contains_text(lines: &[RenderedLine], needle: &str) -> bool {
        // Join each line's spans: markdown rendering may split a phrase across
        // several styled spans, so check the concatenated line text.
        lines.iter().any(|l| {
            let joined: String = l.spans.iter().map(|(t, _)| t.as_str()).collect();
            joined.contains(needle)
        })
    }

    #[test]
    fn unified_path_labels_claude_transcript() {
        let entries = sample_entries();
        let opts = DisplayOptions::default();
        let lines = render_for_display(&entries, &ProviderKind::Claude, &opts);
        let labels = labels(&lines);
        assert!(labels.iter().any(|l| l == "You"), "labels: {:?}", labels);
        assert!(labels.iter().any(|l| l == "Claude"), "labels: {:?}", labels);
        assert!(contains_text(&lines, "hi there"));
        assert!(contains_text(&lines, "hello back"));
    }

    #[test]
    fn unified_path_labels_codex_transcript_not_claude() {
        let entries = sample_entries();
        let opts = DisplayOptions::default();
        let lines = render_for_display(&entries, &ProviderKind::Codex, &opts);
        let labels = labels(&lines);
        // Same fixture, Codex kind: the assistant column is now "Codex", never
        // the hardcoded "Claude" of the old post-select path.
        assert!(labels.iter().any(|l| l == "Codex"), "labels: {:?}", labels);
        assert!(
            !labels.iter().any(|l| l == "Claude"),
            "labels: {:?}",
            labels
        );
        assert!(contains_text(&lines, "hello back"));
    }
}
