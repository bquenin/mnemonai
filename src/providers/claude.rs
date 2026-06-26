use crate::claude::LogEntry;
use crate::conversation_index::delete_conversation;
use crate::error::{AppError, Result};
use crate::history::{self, Conversation, LoaderMessage, ProviderKind};
use crate::tui::viewer;
use std::process::Command;
use std::sync::mpsc::Receiver;

pub struct ClaudeProvider {
    exclude_paths: Vec<String>,
}

impl ClaudeProvider {
    pub fn new(exclude_paths: Vec<String>) -> Self {
        Self { exclude_paths }
    }
}

impl super::Provider for ClaudeProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Claude
    }

    fn name(&self) -> &str {
        "Claude Code"
    }

    fn detect(&self) -> bool {
        // Claude is always available if ~/.claude/projects exists
        history::get_claude_projects_root()
            .map(|p| p.exists())
            .unwrap_or(false)
    }

    fn load_conversations(
        &self,
        show_last: bool,
        debug: Option<crate::cli::DebugLevel>,
    ) -> Result<Vec<Conversation>> {
        history::load_all_conversations(show_last, debug, &self.exclude_paths)
    }

    fn load_conversations_streaming(
        &self,
        show_last: bool,
        debug: Option<crate::cli::DebugLevel>,
    ) -> Receiver<LoaderMessage> {
        history::load_all_conversations_streaming(show_last, debug, self.exclude_paths.clone())
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        viewer::read_log_entries(&conversation.path).map_err(AppError::Io)
    }

    fn resume(&self, conversation: &Conversation, default_args: &[String]) -> Result<()> {
        let project_dir = match &conversation.project_path {
            Some(path) if path.is_dir() => path.clone(),
            Some(path) => {
                // The recorded directory is gone. Claude locates a session by the
                // directory it is launched from, so resuming from anywhere else
                // fails. If this was a worktree of a repo that still exists, move
                // the session into that repo's history and resume there.
                let repo_root = history::resolve_project_dir(path).ok_or_else(|| {
                    AppError::ClaudeExecutionError(format!(
                        "Project directory no longer exists: {}",
                        path.display()
                    ))
                })?;
                rehome_session(conversation, &repo_root)?;
                eprintln!(
                    "Worktree {} is gone; moved this session into {} and resuming there.",
                    path.display(),
                    repo_root.display()
                );
                repo_root
            }
            None => {
                return Err(AppError::ClaudeExecutionError(
                    "Cannot determine project directory for this conversation".to_string(),
                ));
            }
        };

        let mut command = Command::new("claude");
        command.args(["--resume", &conversation.id]);
        command.args(default_args);
        command.current_dir(&project_dir);

        run_claude_command(command)
    }

    fn delete(&self, conversation: &Conversation) -> Result<()> {
        std::fs::remove_file(&conversation.path).map_err(AppError::Io)?;
        delete_conversation(ProviderKind::Claude, &conversation.path);
        Ok(())
    }
}

/// Move a conversation's on-disk footprint — the `<id>.jsonl` transcript and its
/// `<id>/` sidecar directory of tool results and subagent transcripts — out of a
/// torn-down worktree's project directory and into `repo_root`'s project
/// directory, so that `claude --resume`, which locates sessions by launch
/// directory, can find it.
///
/// Moves rather than copies so no orphaned duplicate is left behind, and removes
/// the source project directory afterwards if it is now empty.
fn rehome_session(conversation: &Conversation, repo_root: &std::path::Path) -> Result<()> {
    let src_jsonl = conversation.path.clone();
    let src_dir = src_jsonl
        .parent()
        .ok_or_else(|| {
            AppError::ClaudeExecutionError(
                "Conversation transcript has no parent directory".to_string(),
            )
        })?
        .to_path_buf();

    let dest_dir = history::get_claude_projects_dir(repo_root)?;
    move_session_footprint(&src_jsonl, &src_dir, &dest_dir)
}

/// Move a session's `<id>.jsonl` transcript and its `<id>/` sidecar directory
/// from `src_dir` into `dest_dir`, then remove `src_dir` if it is left empty.
///
/// Skips any destination entry that already exists (e.g. a prior rehome) rather
/// than clobbering it. Kept free of `$HOME`/config lookups so it is unit-testable.
fn move_session_footprint(
    src_jsonl: &std::path::Path,
    src_dir: &std::path::Path,
    dest_dir: &std::path::Path,
) -> Result<()> {
    if dest_dir == src_dir {
        return Ok(()); // already in the target project directory
    }
    std::fs::create_dir_all(dest_dir).map_err(AppError::Io)?;

    // The transcript plus, if present, its sidecar directory (same stem, no
    // extension): `<id>.jsonl` and `<id>/`.
    let sidecar = src_jsonl.with_extension("");
    for src in [src_jsonl.to_path_buf(), sidecar] {
        let Some(name) = src.file_name() else {
            continue;
        };
        if !src.exists() {
            continue;
        }
        let dest = dest_dir.join(name);
        if dest.exists() {
            continue;
        }
        std::fs::rename(&src, &dest).map_err(AppError::Io)?;
    }

    // Drop the now-orphaned worktree project directory if nothing remains;
    // remove_dir fails (and is ignored) when other sessions still live there.
    let _ = std::fs::remove_dir(src_dir);

    Ok(())
}

#[cfg(unix)]
fn run_claude_command(mut command: Command) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let err = command.exec();
    Err(AppError::ClaudeExecutionError(err.to_string()))
}

#[cfg(not(unix))]
fn run_claude_command(mut command: Command) -> Result<()> {
    let status = command
        .status()
        .map_err(|e| AppError::ClaudeExecutionError(e.to_string()))?;

    if !status.success() {
        return Err(AppError::ClaudeExecutionError(format!(
            "claude CLI exited with status {}",
            status
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(name: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&base);
        base
    }

    #[test]
    fn moves_transcript_and_sidecar_and_removes_empty_src() {
        let base = fresh("mnemonai_rehome_move");
        let src_dir = base.join("src-proj");
        let dest_dir = base.join("dest-proj");
        std::fs::create_dir_all(&src_dir).unwrap();
        let id = "11111111-2222-3333-4444-555555555555";
        let jsonl = src_dir.join(format!("{id}.jsonl"));
        std::fs::write(&jsonl, b"transcript").unwrap();
        std::fs::create_dir_all(src_dir.join(id).join("tool-results")).unwrap();
        std::fs::write(src_dir.join(id).join("tool-results").join("a.txt"), b"out").unwrap();

        move_session_footprint(&jsonl, &src_dir, &dest_dir).unwrap();

        assert!(dest_dir.join(format!("{id}.jsonl")).exists());
        assert!(
            dest_dir
                .join(id)
                .join("tool-results")
                .join("a.txt")
                .exists()
        );
        assert!(
            !src_dir.exists(),
            "empty source project dir should be removed"
        );

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn keeps_src_dir_when_other_sessions_remain() {
        let base = fresh("mnemonai_rehome_keep");
        let src_dir = base.join("src-proj");
        let dest_dir = base.join("dest-proj");
        std::fs::create_dir_all(&src_dir).unwrap();
        let id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        let jsonl = src_dir.join(format!("{id}.jsonl"));
        std::fs::write(&jsonl, b"x").unwrap();
        std::fs::write(src_dir.join("other-session.jsonl"), b"y").unwrap();

        move_session_footprint(&jsonl, &src_dir, &dest_dir).unwrap();

        assert!(dest_dir.join(format!("{id}.jsonl")).exists());
        assert!(
            src_dir.exists(),
            "source dir kept when another session remains"
        );
        assert!(src_dir.join("other-session.jsonl").exists());

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn does_not_clobber_existing_destination() {
        let base = fresh("mnemonai_rehome_noclobber");
        let src_dir = base.join("src-proj");
        let dest_dir = base.join("dest-proj");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::create_dir_all(&dest_dir).unwrap();
        let id = "99999999-8888-7777-6666-555555555555";
        let src_jsonl = src_dir.join(format!("{id}.jsonl"));
        std::fs::write(&src_jsonl, b"new").unwrap();
        let dest_jsonl = dest_dir.join(format!("{id}.jsonl"));
        std::fs::write(&dest_jsonl, b"existing").unwrap();

        move_session_footprint(&src_jsonl, &src_dir, &dest_dir).unwrap();

        assert_eq!(
            std::fs::read(&dest_jsonl).unwrap(),
            b"existing",
            "must not clobber an existing destination transcript"
        );
        assert!(
            src_jsonl.exists(),
            "source left intact when destination exists"
        );

        std::fs::remove_dir_all(&base).unwrap();
    }
}
