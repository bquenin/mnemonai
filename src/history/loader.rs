//! Conversation loading and project discovery.
//!
//! This module handles loading conversations from Claude project directories,
//! both synchronously and via streaming for the TUI.

use super::parser::process_conversation_file;
use super::path::{
    decode_project_dir_name, decode_project_dir_name_to_path, format_short_name_from_path,
};
use super::{Conversation, LoaderMessage, Project};
use crate::cli::DebugLevel;
use crate::conversation_index::{LoadedConversation, ProviderCache, fingerprint_from_metadata};
use crate::debug;
use crate::error::{AppError, Result};
use crate::history::ProviderKind;
use crate::providers::LoadOptions;
use rayon::prelude::*;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::SystemTime;

/// Turn one project's loaded entries into finished conversations, injecting the
/// project name/path exactly as before: prefer the cwd parsed from the JSONL,
/// falling back to the decoded project directory name. Consumes `loaded`,
/// dropping full text when the caller asked for the metadata profile.
fn finish_project(
    project_name: &str,
    loaded: Vec<LoadedConversation>,
    include_full_text: bool,
) -> Vec<Conversation> {
    let fallback_path = decode_project_dir_name_to_path(project_name);
    let mut conversations: Vec<Conversation> = loaded
        .into_iter()
        .map(|entry| {
            let mut conv = entry.into_conversation(include_full_text);
            let project_path = conv.cwd.clone().unwrap_or_else(|| fallback_path.clone());
            conv.project_name = Some(format_short_name_from_path(&project_path));
            conv.project_path = Some(project_path);
            conv
        })
        .collect();
    conversations.sort_by_key(|c| std::cmp::Reverse(c.timestamp));
    conversations
}

/// Load conversations from ALL projects globally
pub fn load_all_conversations(
    options: LoadOptions,
    exclude_paths: &[String],
) -> Result<Vec<Conversation>> {
    let debug_level = options.debug;
    let root = super::get_claude_projects_root()?;
    let projects = list_projects(&root, exclude_paths)?;

    debug::info(
        debug_level,
        &format!("Loading global history from {} projects", projects.len()),
    );

    let cache = ProviderCache::load(
        ProviderKind::Claude,
        options.show_last,
        options.include_full_text,
    );
    let failed_projects: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

    // Load conversations from all projects in parallel, keeping each project's
    // entries alive (rather than cloning the fresh ones) so they can be saved
    // from references in a single transaction below.
    let per_project: Vec<(String, Vec<LoadedConversation>)> = projects
        .par_iter()
        .filter_map(|project| {
            let project_dir = root.join(&project.name);
            match load_project_loaded(&project_dir, &cache, options.show_last, debug_level) {
                Ok(loaded) => Some((project.name.clone(), loaded)),
                Err(e) => {
                    debug::warn(
                        debug_level,
                        &format!("Failed to load project {}: {}", project.display_name, e),
                    );
                    failed_projects.lock().unwrap().push(project_dir);
                    None
                }
            }
        })
        .collect();

    // One write transaction for the whole run, instead of one per project
    // racing each other for the SQLite write lock.
    cache.save_fresh(per_project.iter().flat_map(|(_, loaded)| loaded.iter()));

    // Entries no project claimed belong to files that no longer exist — except
    // under a project that failed to load, where we simply don't know.
    let failed = failed_projects.into_inner().unwrap_or_default();
    cache.prune_unclaimed(&failed);

    // Consume into finished conversations with per-project name/path injection.
    let mut all_conversations: Vec<Conversation> = per_project
        .into_iter()
        .flat_map(|(name, loaded)| finish_project(&name, loaded, options.include_full_text))
        .collect();

    // Global sort by timestamp (newest first)
    all_conversations.sort_by_key(|c| std::cmp::Reverse(c.timestamp));

    debug::info(
        debug_level,
        &format!(
            "Total global conversations loaded: {}",
            all_conversations.len()
        ),
    );

    Ok(all_conversations)
}

/// Start loading all conversations in the background
/// Returns a receiver that will receive LoaderMessage updates
pub fn load_all_conversations_streaming(
    options: LoadOptions,
    exclude_paths: Vec<String>,
) -> Receiver<LoaderMessage> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        load_all_streaming_inner(tx, options, &exclude_paths);
    });

    rx
}

fn load_all_streaming_inner(
    tx: Sender<LoaderMessage>,
    options: LoadOptions,
    exclude_paths: &[String],
) {
    let debug_level = options.debug;
    // First, validate that the projects root exists (fatal if not)
    let root = match super::get_claude_projects_root() {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(LoaderMessage::Fatal(e));
            return;
        }
    };

    if !root.exists() {
        let _ = tx.send(LoaderMessage::Fatal(AppError::ProjectsDirNotFound(
            root.display().to_string(),
        )));
        return;
    }

    // List projects (fatal if this fails)
    let projects = match list_projects(&root, exclude_paths) {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.send(LoaderMessage::Fatal(e));
            return;
        }
    };

    debug::info(
        debug_level,
        &format!("Loading global history from {} projects", projects.len()),
    );

    let cache = ProviderCache::load(
        ProviderKind::Claude,
        options.show_last,
        options.include_full_text,
    );
    let failed_projects: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

    // Process projects in parallel and send batches as they complete. Each
    // project persists its own fresh rows from references before its entries are
    // consumed into the batch; on warm starts there are none, so this opens no
    // transaction. Pruning still happens exactly once, after enumeration.
    projects.par_iter().for_each(|project| {
        let project_dir = root.join(&project.name);

        match load_project_loaded(&project_dir, &cache, options.show_last, debug_level) {
            Ok(loaded) => {
                cache.save_fresh(loaded.iter());
                if loaded.is_empty() {
                    return;
                }
                let convs = finish_project(&project.name, loaded, options.include_full_text);
                if convs.is_empty() {
                    return;
                }
                // Send batch, ignore error if receiver dropped
                let _ = tx.send(LoaderMessage::Batch(convs));
            }
            Err(e) => {
                debug::warn(
                    debug_level,
                    &format!("Failed to load project {}: {}", project.display_name, e),
                );
                failed_projects.lock().unwrap().push(project_dir);
                let _ = tx.send(LoaderMessage::ProjectError);
            }
        }
    });

    // Entries no project claimed belong to files that no longer exist — except
    // under a project that failed to load, where we simply don't know.
    let failed = failed_projects.into_inner().unwrap_or_default();
    cache.prune_unclaimed(&failed);

    let _ = tx.send(LoaderMessage::Done);
}

/// List all projects that contain conversation files
fn list_projects(root: &Path, exclude_names: &[String]) -> Result<Vec<Project>> {
    let entries = read_dir(root)?;

    let mut projects: Vec<Project> = entries
        .par_bridge()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            if !path.is_dir() {
                return None;
            }

            // Note: we intentionally do NOT read the directory here to check for
            // conversation files. `load_conversations_with_cache` already scans
            // each project directory once to collect its JSONL files; a project
            // with no (non-agent) conversations simply yields zero conversations
            // there and contributes nothing — including nothing to cache pruning,
            // since it owns no cached entries. Pre-scanning here would read every
            // project directory a second time during startup.
            let name = path.file_name()?.to_string_lossy().to_string();
            // Heuristic decode: convert encoded directory name back to readable path
            // The encoding replaces non-alphanumeric chars (except -) with -
            // So / becomes -, but _ also becomes -, and __ becomes --
            // We convert single dashes to / but preserve double dashes as _
            let display_name = decode_project_dir_name(&name);

            // Skip projects whose display name contains any exclude string
            if exclude_names
                .iter()
                .any(|ex| display_name.contains(ex.as_str()))
            {
                return None;
            }

            let modified = entry
                .metadata()
                .ok()?
                .modified()
                .ok()
                .unwrap_or(SystemTime::UNIX_EPOCH);

            Some(Project {
                name,
                display_name,
                modified,
            })
        })
        .collect();

    // Sort by recently modified
    projects.sort_by_key(|c| std::cmp::Reverse(c.modified));

    Ok(projects)
}

/// Load one project directory into cache-resolved entries, without injecting
/// project info or persisting. The shared [`ProviderCache`] consumes each file's
/// entry on the attempt (cache hit or miss) and claims files it cannot
/// fingerprint, so leftover map entries are exactly the deleted files. The
/// caller persists the fresh entries from references and finishes the
/// conversations.
fn load_project_loaded(
    projects_dir: &Path,
    cache: &ProviderCache,
    show_last: bool,
    debug_level: Option<DebugLevel>,
) -> Result<Vec<LoadedConversation>> {
    // Find all JSONL files and capture metadata in one pass
    let mut files_with_meta = Vec::new();
    let mut skipped_agent_files = 0;

    for entry in read_dir(projects_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str())
                && filename.starts_with("agent-")
            {
                skipped_agent_files += 1;
                debug::debug(debug_level, &format!("Skipping agent file: {}", filename));
                continue;
            }

            let metadata = entry.metadata().ok();
            let modified = metadata
                .as_ref()
                .and_then(|metadata| metadata.modified().ok());
            let fingerprint = metadata.as_ref().map(fingerprint_from_metadata);

            files_with_meta.push((path, modified, fingerprint));
        }
    }

    debug::info(
        debug_level,
        &format!(
            "Found {} conversation files ({} agent files skipped)",
            files_with_meta.len(),
            skipped_agent_files
        ),
    );

    // Sort by modification time (newest first)
    files_with_meta.sort_by_key(|(_, modified, _)| modified.unwrap_or(SystemTime::UNIX_EPOCH));
    files_with_meta.reverse();

    // Process each file (potentially in parallel)
    let loaded: Vec<LoadedConversation> = files_with_meta
        .into_par_iter()
        .filter_map(|(path, modified, fingerprint)| {
            let filename = path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("unknown")
                .to_owned();

            // Consume the cache entry on every attempt so leftovers are exactly
            // the deleted files; a file that failed to stat is claimed too, so
            // it is never mistaken for a deletion.
            let cached = match fingerprint {
                Some(fingerprint) => cache.take_if_fresh(&path, fingerprint),
                None => {
                    cache.claim(&path);
                    None
                }
            };
            if let Some(conversation) = cached {
                debug::debug(
                    debug_level,
                    &format!("Loaded {} from conversation index", filename),
                );
                return Some(LoadedConversation::Cached(conversation));
            }

            match process_conversation_file(path, show_last, modified, debug_level) {
                Ok(Some((conversation, previews))) => {
                    // Only build this message (it embeds the whole preview) when
                    // debug logging is actually enabled.
                    if debug_level.is_some() {
                        debug::debug(
                            debug_level,
                            &format!("Loaded {}: {}", filename, conversation.preview),
                        );
                    }
                    match fingerprint {
                        Some(fingerprint) => Some(LoadedConversation::Fresh {
                            conversation,
                            previews,
                            fingerprint,
                        }),
                        None => Some(LoadedConversation::Cached(conversation)),
                    }
                }
                Ok(None) => None,
                Err(e) => {
                    debug::warn(
                        debug_level,
                        &format!("Error processing {}: {}", filename, e),
                    );
                    None
                }
            }
        })
        .collect();

    debug::info(
        debug_level,
        &format!("Total conversations loaded: {}", loaded.len()),
    );

    Ok(loaded)
}
