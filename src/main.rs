mod claude;
mod cli;
mod config;
mod conversation_index;
mod debug;
mod debug_log;
mod display;
mod error;
mod headless;
mod history;
mod loader;
mod pager;
mod providers;
mod syntax;
mod text_processing;
mod tool_format;
mod tui;

use clap::Parser;
use cli::Args;
use error::{AppError, Result};
use history::LoaderMessage;
use providers::Provider;
use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};

fn main() {
    if let Err(e) = run() {
        match e {
            AppError::SelectionCancelled => {
                // User cancelled, exit silently
                std::process::exit(0);
            }
            // A consumer closing the pipe early (e.g. `mnemonai list | head`) is
            // normal; exit quietly instead of printing a misleading error.
            AppError::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::BrokenPipe => {
                std::process::exit(0);
            }
            _ => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Helper function to resolve a boolean setting by merging CLI flags and config values.
///
/// Priority: enable_flag > disable_flag > config_value > default_value
fn resolve_bool_setting(
    enable_flag: bool,
    disable_flag: bool,
    config_value: Option<bool>,
    default_value: bool,
) -> bool {
    if enable_flag {
        true
    } else if disable_flag {
        false
    } else {
        config_value.unwrap_or(default_value)
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let config = config::load_config()?;

    // Merge CLI arguments with config file settings. CLI takes precedence.
    let display_config = config.display.unwrap_or_default();

    // Extract resume config
    let resume_config = config.resume.unwrap_or_default();
    let default_args = resume_config.default_args.as_deref().unwrap_or(&[]);

    // Use positive names internally for clarity
    let show_tools = resolve_bool_setting(
        args.show_tools,
        args.no_tools,
        display_config.no_tools.map(|b| !b),
        false, // Default: hide tools
    );
    // Map CLI flag to ToolDisplayMode
    // --show-tools → Full, --no-tools → Hidden, default → Hidden
    let tool_display = if args.show_tools {
        tui::ToolDisplayMode::Full
    } else if args.no_tools {
        tui::ToolDisplayMode::Hidden
    } else {
        match display_config.no_tools {
            Some(true) => tui::ToolDisplayMode::Hidden,
            Some(false) => tui::ToolDisplayMode::Full,
            None => tui::ToolDisplayMode::Hidden,
        }
    };
    let show_last = resolve_bool_setting(args.last, args.first, display_config.last, false);
    let use_relative_time = resolve_bool_setting(
        args.relative_time,
        args.absolute_time,
        display_config.relative_time,
        false,
    );
    let show_thinking = resolve_bool_setting(
        args.show_thinking,
        args.hide_thinking,
        display_config.show_thinking,
        false,
    );
    let plain_mode = resolve_bool_setting(args.plain, false, display_config.plain, false);
    let use_pager = resolve_bool_setting(
        args.pager,
        args.no_pager,
        display_config.pager,
        std::io::stdout().is_terminal(),
    );
    let use_global = if args.global {
        true
    } else if args.local {
        false
    } else {
        // New interactive default: scope to the current directory tree. An
        // explicit `local = false` config keeps the previous global startup.
        config.local == Some(false)
    };
    let show_deleted_projects =
        args.show_deleted_projects || display_config.show_deleted_projects.unwrap_or(false);

    // Build provider registry. Headless output must depend only on CLI flags,
    // never the config file, so scripts/skills get stable results regardless of
    // the user's interactive `exclude` setting.
    let exclude_paths = if args.command.is_some() {
        Vec::new()
    } else {
        config.exclude.unwrap_or_default()
    };
    let providers: Vec<Box<dyn Provider>> = vec![
        Box::new(providers::claude::ClaudeProvider::new(exclude_paths)),
        Box::new(providers::codex::CodexProvider::new()),
        Box::new(providers::cursor_agent::CursorAgentProvider::new()),
        Box::new(providers::cursor::CursorProvider::new()),
    ];

    if let Some(ref command) = args.command {
        // Flags only — ignore config-derived show_last / show_deleted_projects so
        // headless output is reproducible across differently-configured machines.
        let settings = headless::HeadlessSettings {
            cli_local: args.local,
            cli_global: args.global,
            show_last: resolve_bool_setting(args.last, args.first, None, false),
            show_deleted_projects: args.show_deleted_projects,
            debug: args.debug,
        };
        return headless::run_command(command, &providers, &settings);
    }

    // Handle --bench-startup flag: time the streaming load headlessly and exit
    if args.bench_startup {
        return bench_startup(&providers, show_last, args.debug);
    }

    // Handle --render flag: render a JSONL file in ledger format and exit
    if let Some(ref render_path) = args.render {
        let display_options = display::DisplayOptions {
            no_tools: !show_tools,
            show_thinking,
            use_pager,
            no_color: args.no_color,
        };
        return display::render_to_terminal(render_path, &display_options);
    }

    // Handle direct file input mode
    if let Some(ref input_file) = args.input_file {
        if !input_file.exists() {
            // A bare word with no path separator or extension (e.g. `dump`,
            // `search`, or a misspelled `list`) was likely meant as a
            // subcommand, which clap routed to the legacy file positional.
            let looks_like_subcommand = input_file
                .to_str()
                .is_some_and(|name| !name.contains(['/', '\\', '.']));
            let message = if looks_like_subcommand {
                format!(
                    "File not found: {} (if you meant a subcommand, see `mnemonai --help`)",
                    input_file.display()
                )
            } else {
                format!("File not found: {}", input_file.display())
            };
            return Err(AppError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                message,
            )));
        }
        if !input_file.is_file() {
            return Err(AppError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Not a file: {}", input_file.display()),
            )));
        }
        tui::run_single_file(
            input_file.clone(),
            use_relative_time,
            tool_display,
            show_thinking,
            &providers,
        )?;
        return Ok(());
    }

    // Handle --show-dir flag (Claude-specific, print directory and exit)
    if args.show_dir {
        if let Ok(current_dir) = std::env::current_dir()
            && let Ok(projects_dir) = history::get_claude_projects_dir(&current_dir)
        {
            println!("{}", projects_dir.display());
        }
        return Ok(());
    }

    // Determine how to load conversations based on mode
    let (conversations, selected_path) = if use_global {
        // Global mode - merge streaming loaders from all providers
        let receivers: Vec<_> = providers
            .iter()
            .map(|p| p.load_conversations_streaming(show_last, args.debug))
            .collect();
        let rx = merge_streaming_loaders(receivers);

        match tui::run_with_loader(
            rx,
            use_relative_time,
            tool_display,
            show_thinking,
            show_deleted_projects,
            &providers,
        )? {
            (tui::Action::Select(path), convs) => (convs, path),
            (tui::Action::Resume(path), convs) => {
                resume_conversation(&convs, &path, &providers, default_args)?;
                return Ok(());
            }
            (tui::Action::Quit, _) => return Err(AppError::SelectionCancelled),
            (tui::Action::Delete(_), _) => unreachable!("Delete is handled internally"),
        }
    } else {
        // Current-directory-tree mode - merge streaming loaders, then filter
        // batches before they enter the TUI.
        let current_dir = std::env::current_dir()?;
        let scope_roots = loader::filter_path_roots(&current_dir)?;
        let receivers: Vec<_> = providers
            .iter()
            .map(|p| p.load_conversations_streaming(show_last, args.debug))
            .collect();
        let rx = loader::filter_loader_messages(merge_streaming_loaders(receivers), scope_roots);

        match tui::run_with_loader(
            rx,
            use_relative_time,
            tool_display,
            show_thinking,
            show_deleted_projects,
            &providers,
        )? {
            (tui::Action::Select(path), convs) => (convs, path),
            (tui::Action::Resume(path), convs) => {
                resume_conversation(&convs, &path, &providers, default_args)?;
                return Ok(());
            }
            (tui::Action::Quit, _) => return Err(AppError::SelectionCancelled),
            (tui::Action::Delete(_), _) => unreachable!("Delete is handled internally"),
        }
    };

    if args.show_path {
        println!("{}", selected_path.display());
        return Ok(());
    }

    if args.show_id {
        let conv = conversations.iter().find(|c| c.path == selected_path);
        let id = conv
            .map(|c| c.id.as_str())
            .or_else(|| selected_path.file_stem().and_then(|stem| stem.to_str()))
            .ok_or_else(|| {
                AppError::ClaudeExecutionError(
                    "Conversation filename is not valid Unicode".to_string(),
                )
            })?;
        println!("{}", id);
        return Ok(());
    }

    if args.resume {
        resume_conversation(&conversations, &selected_path, &providers, default_args)?;
        return Ok(());
    }

    // Log parse errors to debug log if debug mode is enabled
    if args.debug.is_some()
        && let Some(conv) = conversations.iter().find(|c| c.path == selected_path)
    {
        if let Err(e) = debug_log::log_parse_errors(conv) {
            debug::warn(
                args.debug,
                &format!("Failed to write parse errors to log: {}", e),
            );
        } else if !conv.parse_errors.is_empty() {
            debug::info(
                args.debug,
                &format!(
                    "Logged {} parse error(s) to ~/.local/state/mnemonai/debug.log",
                    conv.parse_errors.len()
                ),
            );
        }
    }

    // Display the selected conversation through its owning provider, so
    // non-Claude and SQLite-backed (Cursor) transcripts get the right entries
    // and per-provider labels. Renders via the same viewer-based ledger as the
    // `--render` path. `--plain` maps to a colorless ledger; otherwise the
    // colored crate auto-detects whether stdout is a terminal.
    let display_options = display::DisplayOptions {
        no_tools: !show_tools,
        show_thinking,
        use_pager,
        no_color: plain_mode,
    };

    if let Some(conv) = conversations.iter().find(|c| c.path == selected_path) {
        let entries = match providers.iter().find(|p| p.kind() == conv.provider) {
            Some(provider) => provider.read_entries(conv)?,
            // No registered provider for this kind: fall back to reading the
            // path directly as Claude JSONL.
            None => tui::viewer::read_log_entries(&selected_path).map_err(AppError::Io)?,
        };
        display::display_conversation(&entries, &conv.provider, &display_options)?;
    } else {
        // The selected path had no matching conversation (should not happen for
        // a TUI selection); render the file directly as a Claude transcript.
        display::render_to_terminal(&selected_path, &display_options)?;
    }

    Ok(())
}

/// Drain every provider's streaming loader without the TUI and print timing
/// information. Used to measure startup loading performance (--bench-startup).
fn bench_startup(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug_level: Option<cli::DebugLevel>,
) -> Result<()> {
    use std::time::Instant;

    let overall = Instant::now();

    // Start all providers first (mirrors real startup), then drain each in its
    // own thread so a slow provider doesn't block the others' message streams.
    let receivers: Vec<(String, Receiver<LoaderMessage>)> = providers
        .iter()
        .map(|p| {
            (
                p.name().to_string(),
                p.load_conversations_streaming(show_last, debug_level),
            )
        })
        .collect();

    let handles: Vec<_> = receivers
        .into_iter()
        .map(|(name, rx)| {
            std::thread::spawn(move || {
                let mut first_batch_ms: Option<u128> = None;
                let mut conversations = 0usize;
                let mut batches = 0usize;
                let mut errors = 0usize;
                for msg in rx {
                    match msg {
                        LoaderMessage::Batch(batch) => {
                            batches += 1;
                            conversations += batch.len();
                            first_batch_ms.get_or_insert_with(|| overall.elapsed().as_millis());
                        }
                        LoaderMessage::ProjectError => errors += 1,
                        LoaderMessage::Fatal(_) => {
                            errors += 1;
                            break;
                        }
                        LoaderMessage::Done => break,
                    }
                }
                let done_ms = overall.elapsed().as_millis();
                (
                    name,
                    first_batch_ms,
                    done_ms,
                    conversations,
                    batches,
                    errors,
                )
            })
        })
        .collect();

    let mut results: Vec<_> = handles.into_iter().filter_map(|h| h.join().ok()).collect();
    results.sort_by_key(|r| r.2);

    let total_ms = overall.elapsed().as_millis();
    let total_convs: usize = results.iter().map(|r| r.3).sum();

    println!(
        "{:<18} {:>12} {:>10} {:>8} {:>8} {:>7}",
        "provider", "first batch", "done", "convs", "batches", "errors"
    );
    for (name, first_batch_ms, done_ms, convs, batches, errors) in &results {
        println!(
            "{:<18} {:>12} {:>10} {:>8} {:>8} {:>7}",
            name,
            first_batch_ms
                .map(|ms| format!("{} ms", ms))
                .unwrap_or_else(|| "-".to_string()),
            format!("{} ms", done_ms),
            convs,
            batches,
            errors
        );
    }
    println!(
        "\ntotal: {} conversations in {} ms (all providers done)",
        total_convs, total_ms
    );

    Ok(())
}

/// Merge multiple streaming loader receivers into a single receiver.
/// Each provider streams independently; batches are forwarded immediately.
/// Done is only sent when ALL providers have finished.
/// Fatal errors from individual providers are downgraded to ProjectError
/// so the app continues with other providers.
fn merge_streaming_loaders(receivers: Vec<Receiver<LoaderMessage>>) -> Receiver<LoaderMessage> {
    let (tx, rx) = mpsc::channel();
    let remaining = Arc::new(AtomicUsize::new(receivers.len()));

    for receiver in receivers {
        let tx = tx.clone();
        let remaining = remaining.clone();
        std::thread::spawn(move || {
            for msg in receiver {
                match msg {
                    LoaderMessage::Done => {
                        // Only send Done when all providers have finished
                        if remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                            let _ = tx.send(LoaderMessage::Done);
                        }
                    }
                    LoaderMessage::Fatal(_) => {
                        // Downgrade: one provider failing shouldn't kill the app
                        let _ = tx.send(LoaderMessage::ProjectError);
                        if remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                            let _ = tx.send(LoaderMessage::Done);
                        }
                    }
                    other => {
                        let _ = tx.send(other);
                    }
                }
            }
        });
    }

    rx
}

/// Resume a conversation through the appropriate provider
fn resume_conversation(
    conversations: &[history::Conversation],
    path: &std::path::Path,
    providers: &[Box<dyn Provider>],
    default_args: &[String],
) -> Result<()> {
    let conv = conversations
        .iter()
        .find(|c| c.path == path)
        .ok_or_else(|| {
            AppError::ClaudeExecutionError("Conversation not found for resume".to_string())
        })?;

    let provider = providers
        .iter()
        .find(|p| p.kind() == conv.provider)
        .ok_or_else(|| {
            AppError::ClaudeExecutionError(format!(
                "No provider found for {:?} conversation",
                conv.provider
            ))
        })?;

    provider.resume(conv, default_args)
}
