//! Shared conversation-loading helpers used by both the interactive TUI and the
//! headless commands so they apply identical local and provider filtering.

use crate::cli::{DebugLevel, ProviderFilter};
use crate::error::{AppError, Result};
use crate::history::{Conversation, LoaderMessage, ProviderKind};
use crate::providers::Provider;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;

/// Whether a provider should be included given an optional provider filter.
/// `None` matches every provider.
pub fn provider_filter_matches(filter: Option<ProviderFilter>, kind: &ProviderKind) -> bool {
    filter.is_none_or(|filter| &filter.kind() == kind)
}

/// Load conversations scoped to the current directory tree across all providers.
///
/// The scope is inclusive: a conversation whose recorded cwd or project path is
/// the current directory or any descendant is included. Providers that fail to
/// load are silently skipped. The result is unsorted and unindexed; callers sort
/// by timestamp and assign display indexes.
pub fn load_local(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug: Option<DebugLevel>,
    provider_filter: Option<ProviderFilter>,
) -> Result<Vec<Conversation>> {
    let current_dir = std::env::current_dir().map_err(|err| {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Failed to get current directory: {}", err),
        ))
    })?;

    let roots = filter_path_roots(&current_dir)?;
    load_scoped(providers, show_last, debug, provider_filter, &roots)
}

/// Load conversations scoped to explicit root candidates.
pub fn load_scoped(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug: Option<DebugLevel>,
    provider_filter: Option<ProviderFilter>,
    roots: &[PathBuf],
) -> Result<Vec<Conversation>> {
    let mut conversations = Vec::new();
    for provider in providers {
        if !provider_filter_matches(provider_filter, &provider.kind()) {
            continue;
        }

        if let Ok(mut provider_conversations) = provider.load_conversations(show_last, debug) {
            retain_conversations_in_scope(&mut provider_conversations, roots);
            conversations.extend(provider_conversations);
        }
    }

    Ok(conversations)
}

/// Candidate root forms for a cwd filter: the absolutized literal path and,
/// when it resolves, its canonical (symlink-resolved) form. A conversation path
/// is matched against both because a recorded cwd that no longer exists on disk
/// can't be canonicalized, so it would otherwise fail to match a canonical root
/// even when it is genuinely under the path (e.g. `/tmp` vs `/private/tmp`).
pub fn filter_path_roots(path: &Path) -> Result<Vec<PathBuf>> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };

    let mut roots = Vec::with_capacity(2);
    if let Ok(canonical) = absolute.canonicalize()
        && canonical != absolute
    {
        roots.push(canonical);
    }
    roots.push(absolute);
    Ok(roots)
}

pub fn retain_conversations_in_scope(conversations: &mut Vec<Conversation>, roots: &[PathBuf]) {
    conversations.retain(|conversation| conversation_matches_scope(conversation, roots));
}

pub fn filter_loader_messages(
    rx: Receiver<LoaderMessage>,
    roots: Vec<PathBuf>,
) -> Receiver<LoaderMessage> {
    let (tx, filtered_rx) = mpsc::channel();
    thread::spawn(move || {
        for message in rx {
            match message {
                LoaderMessage::Batch(mut batch) => {
                    retain_conversations_in_scope(&mut batch, &roots);
                    if !batch.is_empty() {
                        let _ = tx.send(LoaderMessage::Batch(batch));
                    }
                }
                other => {
                    let done = matches!(other, LoaderMessage::Done | LoaderMessage::Fatal(_));
                    let _ = tx.send(other);
                    if done {
                        break;
                    }
                }
            }
        }
    });
    filtered_rx
}

pub fn conversation_matches_scope(conversation: &Conversation, roots: &[PathBuf]) -> bool {
    conversation
        .cwd
        .as_ref()
        .into_iter()
        .chain(conversation.project_path.as_ref())
        .any(|path| path_at_or_under(path, roots))
}

fn path_at_or_under(path: &Path, roots: &[PathBuf]) -> bool {
    let canonical = path.canonicalize();
    let candidates = canonical.iter().map(PathBuf::as_path).chain([path]);
    candidates
        .into_iter()
        .any(|candidate| roots.iter().any(|root| candidate.starts_with(root)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Local, TimeZone};

    fn conversation(id: &str, cwd: Option<PathBuf>, project_path: Option<PathBuf>) -> Conversation {
        Conversation {
            path: PathBuf::from(format!("/tmp/{id}.jsonl")),
            index: 0,
            provider: ProviderKind::Codex,
            id: id.to_string(),
            timestamp: Local.timestamp_opt(1_700_000_000, 0).single().unwrap(),
            preview: "preview".to_string(),
            full_text: "full text".to_string(),
            project_name: Some("project".to_string()),
            project_path,
            cwd,
            message_count: 1,
            parse_errors: Vec::new(),
            summary: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
            search_text_lower: None,
            search_topic_end: None,
        }
    }

    #[test]
    fn provider_filter_matches_expected_kinds() {
        assert!(provider_filter_matches(None, &ProviderKind::Cursor));
        assert!(provider_filter_matches(
            Some(ProviderFilter::Codex),
            &ProviderKind::Codex
        ));
        assert!(!provider_filter_matches(
            Some(ProviderFilter::Codex),
            &ProviderKind::Claude
        ));
        assert!(provider_filter_matches(
            Some(ProviderFilter::CursorAgent),
            &ProviderKind::CursorAgent
        ));
    }

    #[test]
    fn scope_matching_includes_root_and_descendants() {
        let root =
            std::env::temp_dir().join(format!("mnemonai-loader-scope-{}", std::process::id()));
        let subdir = root.join("qube/bquenin/mnemonai");
        let other = root.with_file_name(format!(
            "mnemonai-loader-scope-other-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::create_dir_all(&other).unwrap();

        let roots = filter_path_roots(&root).unwrap();
        assert!(conversation_matches_scope(
            &conversation("at-root", Some(root.clone()), None),
            &roots
        ));
        assert!(conversation_matches_scope(
            &conversation("nested", Some(subdir.clone()), None),
            &roots
        ));
        assert!(conversation_matches_scope(
            &conversation("project-path", None, Some(subdir)),
            &roots
        ));
        assert!(!conversation_matches_scope(
            &conversation("outside", Some(other.clone()), Some(other.clone())),
            &roots
        ));

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(other);
    }

    #[test]
    fn scope_matching_keeps_deleted_descendant_paths() {
        let root = std::env::temp_dir().join(format!(
            "mnemonai-loader-deleted-scope-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let gone = root.join("repo/.worktrees/feature");
        let roots = filter_path_roots(&root).unwrap();

        assert!(conversation_matches_scope(
            &conversation("gone", Some(gone.clone()), Some(gone)),
            &roots
        ));

        let _ = std::fs::remove_dir_all(root);
    }
}
