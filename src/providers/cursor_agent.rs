use crate::claude::{
    AssistantMessage, ContentBlock, LogEntry, UserContent, UserMessage, extract_text_from_blocks,
};
use crate::cli::DebugLevel;
use crate::conversation_index::{
    CachedFileConversation, SourceFingerprint, delete_conversation, fingerprint_from_metadata,
    load_provider_cache, save_conversations,
};
use crate::debug;
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, ParseError, ProviderKind, format_short_name_from_path,
};
use chrono::{DateTime, FixedOffset, Local};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver};
use std::time::SystemTime;

pub struct CursorAgentProvider {
    projects_root: PathBuf,
}

struct AgentProject {
    project_dir_name: String,
    workspace_path: Option<PathBuf>,
    transcript_files: Vec<PathBuf>,
    modified: SystemTime,
}

#[derive(Debug, Default, Deserialize)]
struct CursorAgentTranscriptRecord {
    role: String,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    message: CursorAgentTranscriptMessage,
}

#[derive(Debug, Default, Deserialize)]
struct CursorAgentTranscriptMessage {
    #[serde(default)]
    content: Vec<CursorAgentTranscriptBlock>,
}

#[derive(Debug, Default, Deserialize)]
struct CursorAgentTranscriptBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    source: Option<Value>,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default)]
    id: Option<String>,
}

struct ParsedTranscriptLine {
    entry: LogEntry,
    text_for_search: Option<String>,
    text_for_preview: Option<String>,
    timestamp: Option<DateTime<FixedOffset>>,
    counts_as_message: bool,
}

enum ConversationLoad {
    Cached(Conversation),
    Fresh(Conversation, SourceFingerprint),
}

impl ConversationLoad {
    fn into_conversation(self) -> Conversation {
        match self {
            Self::Cached(conversation) | Self::Fresh(conversation, _) => conversation,
        }
    }
}

impl CursorAgentProvider {
    pub fn new() -> Self {
        let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
        Self {
            projects_root: home.join(".cursor").join("projects"),
        }
    }

    fn list_projects(&self) -> Result<Vec<AgentProject>> {
        if !self.projects_root.exists() {
            return Ok(Vec::new());
        }

        let chats_dir = self
            .projects_root
            .parent()
            .map(|cursor_dir| cursor_dir.join("chats"));

        let mut projects = Vec::new();
        for entry in fs::read_dir(&self.projects_root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let transcripts_dir = path.join("agent-transcripts");
            if !transcripts_dir.is_dir() {
                continue;
            }

            let transcript_files = collect_transcript_files(&transcripts_dir);
            if transcript_files.is_empty() {
                continue;
            }

            let modified = transcript_files
                .iter()
                .filter_map(|path| file_modified_time(path))
                .max()
                .unwrap_or(SystemTime::UNIX_EPOCH);

            let project_dir_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .to_string();

            let workspace_path =
                resolve_workspace_path(&project_dir_name, chats_dir.as_deref());

            projects.push(AgentProject {
                project_dir_name,
                workspace_path,
                transcript_files,
                modified,
            });
        }

        projects.sort_by(|a, b| b.modified.cmp(&a.modified));
        Ok(projects)
    }

    /// Load one project's transcripts. Returns the conversations plus clones of
    /// the freshly parsed ones (cache misses) so the caller can persist them in
    /// a single batch at the end of the run.
    fn load_project_conversations(
        &self,
        project: &AgentProject,
        show_last: bool,
        debug_level: Option<DebugLevel>,
        cache: &Mutex<HashMap<PathBuf, CachedFileConversation>>,
    ) -> (Vec<Conversation>, Vec<(Conversation, SourceFingerprint)>) {
        let workspace_path = project.workspace_path.clone();
        let project_dir_name = project.project_dir_name.clone();
        let loaded: Vec<ConversationLoad> = project
            .transcript_files
            .par_iter()
            .filter_map(|path| {
                let filename = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let fingerprint = file_fingerprint(path);

                if let Some(fingerprint) = fingerprint
                    && let Some(conversation) = cache
                        .lock()
                        .ok()
                        .and_then(|mut cache| cache.remove(path))
                        .and_then(|cached| cached.into_conversation_if_fresh(fingerprint))
                {
                    debug::debug(
                        debug_level,
                        &format!("Loaded Cursor Agent transcript {} from index", filename),
                    );
                    return Some(ConversationLoad::Cached(conversation));
                }

                match process_transcript_file(
                    path.clone(),
                    show_last,
                    workspace_path.clone(),
                    project_dir_name.clone(),
                    debug_level,
                ) {
                    Ok(Some(conversation)) => {
                        debug::debug(
                            debug_level,
                            &format!(
                                "Loaded Cursor Agent transcript {}: {}",
                                filename, conversation.preview
                            ),
                        );
                        match fingerprint {
                            Some(fingerprint) => {
                                Some(ConversationLoad::Fresh(conversation, fingerprint))
                            }
                            None => Some(ConversationLoad::Cached(conversation)),
                        }
                    }
                    Ok(None) => None,
                    Err(err) => {
                        debug::warn(
                            debug_level,
                            &format!(
                                "Failed to process Cursor Agent transcript {}: {}",
                                filename, err
                            ),
                        );
                        None
                    }
                }
            })
            .collect();

        let fresh: Vec<(Conversation, SourceFingerprint)> = loaded
            .iter()
            .filter_map(|loaded| match loaded {
                ConversationLoad::Fresh(conversation, fingerprint) => {
                    Some((conversation.clone(), *fingerprint))
                }
                ConversationLoad::Cached(_) => None,
            })
            .collect();

        let mut conversations: Vec<Conversation> = loaded
            .into_iter()
            .map(ConversationLoad::into_conversation)
            .collect();

        conversations.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        for (idx, conversation) in conversations.iter_mut().enumerate() {
            conversation.index = idx;
        }
        (conversations, fresh)
    }
}

impl super::Provider for CursorAgentProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::CursorAgent
    }

    fn name(&self) -> &str {
        "Cursor Agent CLI"
    }

    fn detect(&self) -> bool {
        self.projects_root.exists()
    }

    fn load_conversations(
        &self,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) -> Result<Vec<Conversation>> {
        let projects = self.list_projects()?;
        if projects.is_empty() {
            return Ok(Vec::new());
        }

        let cache = Mutex::new(load_provider_cache(ProviderKind::CursorAgent, show_last));
        let mut all_fresh: Vec<(Conversation, SourceFingerprint)> = Vec::new();
        let mut conversations: Vec<Conversation> = Vec::new();
        for project in &projects {
            let (convs, fresh) =
                self.load_project_conversations(project, show_last, debug_level, &cache);
            conversations.extend(convs);
            all_fresh.extend(fresh);
        }

        save_conversations(
            ProviderKind::CursorAgent,
            show_last,
            all_fresh
                .iter()
                .map(|(conv, fingerprint)| (conv, *fingerprint)),
        );

        conversations.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        for (idx, conversation) in conversations.iter_mut().enumerate() {
            conversation.index = idx;
        }

        Ok(conversations)
    }

    fn load_conversations_streaming(
        &self,
        show_last: bool,
        debug_level: Option<DebugLevel>,
    ) -> Receiver<LoaderMessage> {
        let (tx, rx) = mpsc::channel();
        let projects_root = self.projects_root.clone();

        std::thread::spawn(move || {
            let provider = CursorAgentProvider { projects_root };
            let projects = match provider.list_projects() {
                Ok(projects) => projects,
                Err(_) => {
                    let _ = tx.send(LoaderMessage::Done);
                    return;
                }
            };

            let cache = Mutex::new(load_provider_cache(ProviderKind::CursorAgent, show_last));
            let mut all_fresh: Vec<(Conversation, SourceFingerprint)> = Vec::new();
            for project in &projects {
                let (conversations, fresh) =
                    provider.load_project_conversations(project, show_last, debug_level, &cache);
                all_fresh.extend(fresh);
                if !conversations.is_empty() {
                    let _ = tx.send(LoaderMessage::Batch(conversations));
                }
            }

            // All batches are on the channel; persist newly parsed transcripts
            // in one write transaction instead of one per project.
            save_conversations(
                ProviderKind::CursorAgent,
                show_last,
                all_fresh
                    .iter()
                    .map(|(conv, fingerprint)| (conv, *fingerprint)),
            );

            let _ = tx.send(LoaderMessage::Done);
        });

        rx
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        let file = File::open(&conversation.path)?;
        let reader = BufReader::new(file);
        let workspace_path = conversation.project_path.as_deref();
        let mut entries = Vec::new();

        for (line_idx, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }

            match parse_transcript_line(&line, line_idx + 1, workspace_path) {
                Ok(Some(parsed)) => entries.push(parsed.entry),
                Ok(None) => {}
                Err(err) => {
                    return Err(AppError::ClaudeExecutionError(format!(
                        "Failed to parse Cursor Agent transcript {} line {}: {}",
                        conversation.path.display(),
                        line_idx + 1,
                        err
                    )));
                }
            }
        }

        Ok(entries)
    }

    fn resume(&self, conversation: &Conversation, _default_args: &[String]) -> Result<()> {
        let workspace_path = if let Some(path) = conversation
            .project_path
            .as_ref()
            .or(conversation.cwd.as_ref())
            .filter(|p| p.is_dir())
        {
            path.clone()
        } else {
            let encoded_dir_name = extract_project_dir_name(&conversation.path).ok_or_else(|| {
                AppError::ClaudeExecutionError(
                    "Cannot determine project directory for this Cursor Agent conversation"
                        .to_string(),
                )
            })?;
            let chats_dir = self
                .projects_root
                .parent()
                .map(|cursor_dir| cursor_dir.join("chats"));
            resolve_workspace_path(&encoded_dir_name, chats_dir.as_deref()).ok_or_else(|| {
                AppError::ClaudeExecutionError(
                    "Cannot determine workspace path for this Cursor Agent conversation"
                        .to_string(),
                )
            })?
        };

        let mut command = build_resume_command(&workspace_path, &conversation.id)?;
        run_cursor_agent_command(&mut command)
    }

    fn delete(&self, conversation: &Conversation) -> Result<()> {
        if let Some(transcript_dir) = transcript_parent_dir(&conversation.path, &conversation.id) {
            if transcript_dir.exists() {
                fs::remove_dir_all(transcript_dir)?;
                delete_conversation(ProviderKind::CursorAgent, &conversation.path);
                return Ok(());
            }
        }

        fs::remove_file(&conversation.path)?;
        delete_conversation(ProviderKind::CursorAgent, &conversation.path);
        Ok(())
    }
}

fn process_transcript_file(
    path: PathBuf,
    show_last: bool,
    workspace_path: Option<PathBuf>,
    project_dir_name: String,
    debug_level: Option<DebugLevel>,
) -> Result<Option<Conversation>> {
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let mut all_parts = Vec::new();
    let mut preview_parts = Vec::new();
    let mut message_count = 0;
    let mut parse_errors = Vec::new();
    let mut first_timestamp: Option<DateTime<FixedOffset>> = None;
    let mut last_timestamp: Option<DateTime<FixedOffset>> = None;

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                parse_errors.push(ParseError {
                    line_number: line_idx + 1,
                    line_content: String::new(),
                    error_message: err.to_string(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        match parse_transcript_line(&line, line_idx + 1, workspace_path.as_deref()) {
            Ok(Some(parsed)) => {
                if let Some(ts) = parsed.timestamp {
                    if first_timestamp.is_none() {
                        first_timestamp = Some(ts);
                    }
                    last_timestamp = Some(ts);
                }

                if let Some(text) = parsed.text_for_search {
                    all_parts.push(text);
                }
                if let Some(text) = parsed.text_for_preview {
                    preview_parts.push(text);
                }
                if parsed.counts_as_message {
                    message_count += 1;
                }
            }
            Ok(None) => {}
            Err(err) => {
                debug::warn(
                    debug_level,
                    &format!(
                        "Cursor Agent parse error in {} at line {}: {}",
                        path.display(),
                        line_idx + 1,
                        err
                    ),
                );
                parse_errors.push(ParseError {
                    line_number: line_idx + 1,
                    line_content: line,
                    error_message: err.to_string(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
            }
        }
    }

    if preview_parts.is_empty() {
        return Ok(None);
    }

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

    let preview = normalize_whitespace(&preview);
    let full_text = normalize_whitespace(&all_parts.join(" "));
    let timestamp = last_timestamp
        .map(|ts| ts.with_timezone(&Local))
        .or_else(|| file_modified_time(&path).map(DateTime::<Local>::from))
        .unwrap_or_else(Local::now);
    let duration_minutes = match (first_timestamp, last_timestamp) {
        (Some(first), Some(last)) => {
            let duration = last.signed_duration_since(first);
            (duration.num_minutes() > 0).then_some(duration.num_minutes() as u64)
        }
        _ => None,
    };
    let id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string();
    let project_name = workspace_path
        .as_ref()
        .map(|workspace| format_short_name_from_path(workspace))
        .or_else(|| {
            let last_segment = project_dir_name.rsplit('-').next()?;
            (!last_segment.is_empty()).then(|| last_segment.to_string())
        });

    Ok(Some(Conversation {
        path,
        index: 0,
        provider: ProviderKind::CursorAgent,
        id,
        timestamp,
        preview,
        full_text,
        project_name,
        project_path: workspace_path.clone(),
        cwd: workspace_path,
        message_count,
        parse_errors,
        summary: None,
        model: None,
        total_tokens: 0,
        duration_minutes,
        search_text_lower: None,
        search_topic_end: None,
    }))
}

fn parse_transcript_line(
    line: &str,
    line_idx: usize,
    workspace_path: Option<&Path>,
) -> Result<Option<ParsedTranscriptLine>> {
    let record: CursorAgentTranscriptRecord = serde_json::from_str(line)?;
    let timestamp = record
        .timestamp
        .as_deref()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
    let timestamp_str = record.timestamp.unwrap_or_default();

    let blocks: Vec<ContentBlock> = record
        .message
        .content
        .iter()
        .enumerate()
        .filter_map(|(block_idx, block)| block_to_content_block(block, line_idx, block_idx))
        .collect();

    if blocks.is_empty() {
        return Ok(None);
    }

    let text = normalize_whitespace(&extract_text_from_blocks(&blocks));
    let summary = if text.is_empty() {
        summarize_blocks_for_preview(&record.message.content)
    } else {
        Some(text.clone())
    };
    let role = record.role.to_lowercase();
    let entry = match role.as_str() {
        "user" => LogEntry::User {
            message: UserMessage {
                role: "user".to_string(),
                content: UserContent::Blocks(blocks),
            },
            timestamp: timestamp_str,
            uuid: None,
            cwd: workspace_path.map(|path| path.to_string_lossy().to_string()),
        },
        "assistant" => LogEntry::Assistant {
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: blocks,
                model: None,
                usage: None,
                id: None,
            },
            timestamp: timestamp_str,
            uuid: None,
        },
        _ => return Ok(None),
    };

    Ok(Some(ParsedTranscriptLine {
        entry,
        text_for_search: summary.clone(),
        text_for_preview: summary,
        timestamp,
        counts_as_message: true,
    }))
}

fn block_to_content_block(
    block: &CursorAgentTranscriptBlock,
    line_idx: usize,
    block_idx: usize,
) -> Option<ContentBlock> {
    match block.block_type.as_str() {
        "text" => block
            .text
            .as_ref()
            .map(|text| ContentBlock::Text { text: text.clone() }),
        "tool_use" => Some(ContentBlock::ToolUse {
            id: block
                .id
                .clone()
                .unwrap_or_else(|| format!("cursor-agent-{}-{}", line_idx, block_idx)),
            name: block.name.clone().unwrap_or_else(|| "tool".to_string()),
            input: block.input.clone().unwrap_or(Value::Null),
        }),
        "tool_result" => Some(ContentBlock::ToolResult {
            tool_use_id: block
                .tool_use_id
                .clone()
                .unwrap_or_else(|| format!("cursor-agent-result-{}-{}", line_idx, block_idx)),
            content: block.content.clone(),
        }),
        "thinking" => Some(ContentBlock::Thinking {
            thinking: block.thinking.clone().unwrap_or_default(),
            signature: block
                .signature
                .clone()
                .unwrap_or_else(|| format!("cursor-agent-thinking-{}-{}", line_idx, block_idx)),
        }),
        "image" => block.source.as_ref().map(|source| ContentBlock::Image {
            source: source.clone(),
        }),
        _ => None,
    }
}

fn summarize_blocks_for_preview(blocks: &[CursorAgentTranscriptBlock]) -> Option<String> {
    let summary = blocks
        .iter()
        .filter_map(|block| match block.block_type.as_str() {
            "tool_use" => block.name.as_ref().map(|name| format!("Tool: {}", name)),
            "thinking" => Some("Thinking".to_string()),
            _ => None,
        })
        .take(3)
        .collect::<Vec<_>>()
        .join(" ... ");

    (!summary.is_empty()).then_some(summary)
}

fn collect_transcript_files(transcripts_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = match fs::read_dir(transcripts_dir) {
        Ok(entries) => entries,
        Err(_) => return files,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
            files.push(path);
            continue;
        }

        if !path.is_dir() {
            continue;
        }

        let default_path = path.join(format!("{}.jsonl", entry.file_name().to_string_lossy()));
        if default_path.is_file() {
            files.push(default_path);
            continue;
        }

        let nested_entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for nested in nested_entries.flatten() {
            let nested_path = nested.path();
            if nested_path.is_file() && nested_path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(nested_path);
                break;
            }
        }
    }

    files.sort_by_key(|path| file_modified_time(path).unwrap_or(SystemTime::UNIX_EPOCH));
    files.reverse();
    files
}

/// Extracts the encoded project directory name from a transcript file path.
///
/// Transcript paths look like:
/// `~/.cursor/projects/<encoded_dir>/agent-transcripts/<chatId>/<chatId>.jsonl`
fn extract_project_dir_name(transcript_path: &Path) -> Option<String> {
    for ancestor in transcript_path.ancestors() {
        if ancestor.file_name().and_then(|n| n.to_str()) == Some("agent-transcripts") {
            return ancestor
                .parent()?
                .file_name()?
                .to_str()
                .map(|s| s.to_string());
        }
    }
    None
}

/// Reconstructs the workspace path from the encoded project directory name.
///
/// Cursor encodes workspace paths by replacing every non-alphanumeric character
/// with `-`, collapsing runs, and trimming edges. This function reverses the
/// encoding via DFS over hyphen positions, pruning branches where the path
/// prefix doesn't exist on disk. If multiple candidates survive, the MD5 hash
/// of `~/.cursor/chats/` directories is used as a tiebreaker.
fn resolve_workspace_path(encoded_dir_name: &str, chats_dir: Option<&Path>) -> Option<PathBuf> {
    if encoded_dir_name.is_empty() {
        return None;
    }
    let segments: Vec<&str> = encoded_dir_name.split('-').collect();

    let mut candidates = Vec::new();
    dfs_resolve(&segments, 0, String::from("/"), &mut candidates);

    match candidates.len() {
        0 => None,
        1 => Some(PathBuf::from(candidates.into_iter().next().unwrap())),
        _ => {
            // MD5 tiebreaker: check which candidate has a matching chats directory
            if let Some(chats_dir) = chats_dir.filter(|d| d.is_dir()) {
                for candidate in &candidates {
                    let hash = format!("{:x}", md5::compute(candidate.as_bytes()));
                    if chats_dir.join(&hash).is_dir() {
                        return Some(PathBuf::from(candidate));
                    }
                }
            }
            // Fall back to the longest path (most specific)
            candidates.sort_by(|a, b| b.len().cmp(&a.len()));
            Some(PathBuf::from(candidates.into_iter().next().unwrap()))
        }
    }
}

/// DFS over hyphen positions, trying each `-` as either a path separator (`/`)
/// or a literal character kept in the current path component. Also tries `.`
/// and ` ` since those are also encoded as `-`.
///
/// Prunes branches where the accumulated path prefix doesn't exist on disk.
fn dfs_resolve(segments: &[&str], idx: usize, current: String, results: &mut Vec<String>) {
    if idx >= segments.len() {
        let path = PathBuf::from(&current);
        if path.exists() {
            results.push(current);
        }
        return;
    }

    if results.len() >= 8 {
        return;
    }

    let segment = segments[idx];
    let is_first = idx == 0;
    let is_last = idx == segments.len() - 1;

    if is_first {
        let candidate = format!("/{}", segment);
        if component_prefix_viable(&candidate, is_last) {
            dfs_resolve(segments, idx + 1, candidate, results);
        }
        return;
    }

    // Option 1: `-` was a `/` — start a new path component
    let with_slash = format!("{}/{}", current, segment);
    if component_prefix_viable(&with_slash, is_last) {
        dfs_resolve(segments, idx + 1, with_slash, results);
    }

    // Option 2: `-` was a `.` — common for hidden dirs and file extensions
    let with_dot = format!("{}.{}", current, segment);
    if component_prefix_viable(&with_dot, is_last) {
        dfs_resolve(segments, idx + 1, with_dot, results);
    }

    // Option 3: `-` was a ` ` (space)
    let with_space = format!("{} {}", current, segment);
    if component_prefix_viable(&with_space, is_last) {
        dfs_resolve(segments, idx + 1, with_space, results);
    }

    // Option 4: `-` was a literal `-` in the directory/file name
    let with_hyphen = format!("{}-{}", current, segment);
    if component_prefix_viable(&with_hyphen, is_last) {
        dfs_resolve(segments, idx + 1, with_hyphen, results);
    }

    // Option 5: `-` was a `/` followed by a `.` — hidden directory
    let with_hidden = format!("{}/.{}", current, segment);
    if component_prefix_viable(&with_hidden, is_last) {
        dfs_resolve(segments, idx + 1, with_hidden, results);
    }
}

/// Checks whether this accumulated path prefix could lead to a valid final path.
///
/// For the last segment, the full path must exist. For intermediate segments,
/// we check if the parent directory has any entry starting with the current
/// file name component — this handles partial names like `/tmp/my-proj` where
/// the final directory is `/tmp/my-project`.
fn component_prefix_viable(prefix: &str, is_last: bool) -> bool {
    let path = PathBuf::from(prefix);
    if is_last {
        return path.exists();
    }
    if path.is_dir() {
        return true;
    }
    // The path doesn't exist as a directory yet — it might be a partial component
    // name. Check if the parent dir has entries starting with this prefix.
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(name_prefix) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if !parent.is_dir() {
        return false;
    }
    let Ok(entries) = fs::read_dir(parent) else {
        return false;
    };
    // Exclude exact matches: if the only entry is the prefix itself, there's no
    // longer name to continue building. Exact-match directories are already
    // handled by the `path.is_dir()` check above.
    entries
        .filter_map(|e| e.ok())
        .any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(name_prefix) && name != name_prefix)
        })
}

fn file_modified_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

fn file_fingerprint(path: &Path) -> Option<SourceFingerprint> {
    fs::metadata(path)
        .ok()
        .map(|metadata| fingerprint_from_metadata(&metadata))
}

fn transcript_parent_dir(path: &Path, conversation_id: &str) -> Option<PathBuf> {
    let parent = path.parent()?;
    let parent_name = parent.file_name()?.to_str()?;
    let grandparent = parent.parent()?;
    let grandparent_name = grandparent.file_name()?.to_str()?;
    if parent_name == conversation_id && grandparent_name == "agent-transcripts" {
        Some(parent.to_path_buf())
    } else {
        None
    }
}

fn build_resume_command(workspace_path: &Path, chat_id: &str) -> Result<Command> {
    let mut command = if command_exists("cursor-agent") {
        Command::new("cursor-agent")
    } else if command_exists("agent") {
        Command::new("agent")
    } else if command_exists("cursor") {
        let mut command = Command::new("cursor");
        command.arg("agent");
        command
    } else {
        return Err(AppError::ClaudeExecutionError(
            "Could not find a Cursor Agent CLI executable (`cursor-agent`, `agent`, or `cursor`)"
                .to_string(),
        ));
    };

    command
        .arg("--resume")
        .arg(chat_id)
        .arg("--workspace")
        .arg(workspace_path)
        .current_dir(workspace_path);

    Ok(command)
}

fn command_exists(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&paths).any(|dir| dir.join(name).is_file())
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(unix)]
fn run_cursor_agent_command(command: &mut Command) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let err = command.exec();
    Err(AppError::ClaudeExecutionError(err.to_string()))
}

#[cfg(not(unix))]
fn run_cursor_agent_command(command: &mut Command) -> Result<()> {
    let status = command
        .status()
        .map_err(|err| AppError::ClaudeExecutionError(err.to_string()))?;

    if !status.success() {
        return Err(AppError::ClaudeExecutionError(format!(
            "Cursor Agent CLI exited with status {}",
            status
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mnemonai-cursor-agent-{}-{}-{}",
                name,
                std::process::id(),
                unique
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn parse_transcript_line_converts_tool_calls() {
        let line = r#"{"role":"assistant","message":{"content":[{"type":"text","text":"Checking now"},{"type":"tool_use","name":"Shell","input":{"command":"git status"}}]}}"#;

        let parsed = parse_transcript_line(line, 1, None).unwrap().unwrap();
        match parsed.entry {
            LogEntry::Assistant { message, .. } => {
                assert_eq!(message.content.len(), 2);
                match &message.content[1] {
                    ContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(name, "Shell");
                        assert!(!id.is_empty());
                        assert_eq!(input["command"], "git status");
                    }
                    other => panic!("expected tool use, got {:?}", other),
                }
            }
            other => panic!("expected assistant entry, got {:?}", other),
        }
        assert_eq!(parsed.text_for_preview.as_deref(), Some("Checking now"));
    }

    #[test]
    fn process_transcript_file_builds_conversation_metadata() {
        let temp = TestTempDir::new("conversation");
        let workspace_path = temp.path.join("workspace");
        fs::create_dir_all(&workspace_path).unwrap();

        let transcript_path = temp.path.join("chat-123.jsonl");
        let mut file = File::create(&transcript_path).unwrap();
        writeln!(
            file,
            r#"{{"role":"user","timestamp":"2026-04-24T20:00:00Z","message":{{"content":[{{"type":"text","text":"hello from cursor agent"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"role":"assistant","timestamp":"2026-04-24T20:02:00Z","message":{{"content":[{{"type":"text","text":"hi there"}},{{"type":"tool_use","name":"Shell","input":{{"command":"pwd"}}}}]}}}}"#
        )
        .unwrap();

        let conversation = process_transcript_file(
            transcript_path,
            false,
            Some(workspace_path.clone()),
            "test-workspace".to_string(),
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(conversation.provider, ProviderKind::CursorAgent);
        assert_eq!(conversation.id, "chat-123");
        assert_eq!(conversation.project_path.as_ref(), Some(&workspace_path));
        assert_eq!(conversation.cwd.as_ref(), Some(&workspace_path));
        assert_eq!(conversation.project_name.as_deref(), Some("workspace"));
        assert!(conversation.preview.contains("hello from cursor agent"));
        assert!(conversation.preview.contains("hi there"));
        assert!(conversation.full_text.contains("hello from cursor agent"));
        assert_eq!(conversation.message_count, 2);
        assert_eq!(conversation.duration_minutes, Some(2));
    }

    #[test]
    fn delete_prefers_transcript_directory() {
        let temp = TestTempDir::new("delete");
        let transcript_dir = temp.path.join("agent-transcripts").join("chat-123");
        fs::create_dir_all(transcript_dir.join("subagents")).unwrap();
        fs::write(transcript_dir.join("chat-123.jsonl"), "[]").unwrap();
        fs::write(transcript_dir.join("subagents").join("worker.jsonl"), "[]").unwrap();

        let provider = CursorAgentProvider {
            projects_root: temp.path.clone(),
        };
        let conversation = Conversation {
            path: transcript_dir.join("chat-123.jsonl"),
            index: 0,
            provider: ProviderKind::CursorAgent,
            id: "chat-123".to_string(),
            timestamp: Local::now(),
            preview: String::new(),
            full_text: String::new(),
            project_name: None,
            project_path: None,
            cwd: None,
            message_count: 0,
            parse_errors: Vec::new(),
            summary: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
            search_text_lower: None,
            search_topic_end: None,
        };

        provider.delete(&conversation).unwrap();
        assert!(!transcript_dir.exists());
    }

    #[test]
    fn extract_project_dir_name_from_transcript_path() {
        let path = PathBuf::from(
            "/home/user/.cursor/projects/Users-me-myproject/agent-transcripts/chat-1/chat-1.jsonl",
        );
        assert_eq!(
            extract_project_dir_name(&path).as_deref(),
            Some("Users-me-myproject")
        );
    }

    #[test]
    fn extract_project_dir_name_returns_none_for_unrelated_path() {
        let path = PathBuf::from("/tmp/random/file.jsonl");
        assert_eq!(extract_project_dir_name(&path), None);
    }

    #[test]
    fn dfs_resolve_finds_path_with_all_slashes() {
        // /tmp always exists on macOS/Linux
        if !PathBuf::from("/tmp").is_dir() {
            return;
        }
        let segments = vec!["tmp"];
        let mut results = Vec::new();
        dfs_resolve(&segments, 0, String::from("/"), &mut results);
        assert!(
            results.contains(&"/tmp".to_string()),
            "DFS should find /tmp in {:?}",
            results
        );
    }

    struct ShallowTempDir {
        path: PathBuf,
    }

    impl ShallowTempDir {
        fn under_tmp(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                PathBuf::from("/tmp").join(format!("mnemonai-{}-{}-{}", name, std::process::id(), unique));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for ShallowTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn dfs_resolve_handles_hyphen_in_name() {
        let temp = ShallowTempDir::under_tmp("dfs-hyphen");
        let nested = temp.path.join("sub").join("a-b");
        fs::create_dir_all(&nested).unwrap();

        let parent_name = temp.path.file_name().unwrap().to_str().unwrap();
        let encoded = format!("tmp-{}-sub-a-b", parent_name);
        let result = resolve_workspace_path(&encoded, None);

        assert_eq!(
            result,
            Some(nested),
            "DFS should reconstruct path with hyphenated directory name"
        );
    }

    #[test]
    fn resolve_workspace_path_returns_none_for_nonexistent() {
        assert_eq!(
            resolve_workspace_path("zzznonexistent-path-that-does-not-exist", None),
            None
        );
    }

    #[test]
    fn resolve_workspace_private_tmp() {
        if !PathBuf::from("/private/tmp").is_dir() {
            return;
        }
        let result = resolve_workspace_path("private-tmp", None);
        assert_eq!(result, Some(PathBuf::from("/private/tmp")));
    }

    #[test]
    fn project_name_falls_back_to_dir_name_segment() {
        let temp = TestTempDir::new("project-name-fallback");
        let transcript_path = temp.path.join("chat-456.jsonl");
        let mut file = File::create(&transcript_path).unwrap();
        writeln!(
            file,
            r#"{{"role":"user","timestamp":"2026-04-24T20:00:00Z","message":{{"content":[{{"type":"text","text":"hello"}}]}}}}"#
        )
        .unwrap();

        let conversation = process_transcript_file(
            transcript_path,
            false,
            None,
            "Users-me-myproject".to_string(),
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(conversation.project_name.as_deref(), Some("myproject"));
    }

    #[test]
    fn dfs_resolve_handles_dot_in_name() {
        let temp = ShallowTempDir::under_tmp("dfs-dot");
        let dotfile = temp.path.join("my.config");
        fs::create_dir_all(&dotfile).unwrap();

        let parent_name = temp.path.file_name().unwrap().to_str().unwrap();
        let encoded = format!("tmp-{}-my-config", parent_name);
        let result = resolve_workspace_path(&encoded, None);

        assert_eq!(
            result,
            Some(dotfile),
            "DFS should reconstruct path with dot-separated name"
        );
    }

    #[test]
    fn md5_tiebreaker_selects_correct_candidate() {
        let temp = ShallowTempDir::under_tmp("md5-tiebreaker");

        // Create two candidates that the DFS would both find
        let candidate_a = temp.path.join("x").join("y");
        let candidate_b = temp.path.join("x-y");
        fs::create_dir_all(&candidate_a).unwrap();
        fs::create_dir_all(&candidate_b).unwrap();

        // Create a fake chats dir with the MD5 of candidate_b
        let chats_dir = temp.path.join("chats");
        let hash_b = format!("{:x}", md5::compute(candidate_b.to_string_lossy().as_bytes()));
        fs::create_dir_all(chats_dir.join(&hash_b)).unwrap();

        let parent_name = temp.path.file_name().unwrap().to_str().unwrap();
        let encoded = format!("tmp-{}-x-y", parent_name);
        let result = resolve_workspace_path(&encoded, Some(chats_dir.as_path()));

        assert_eq!(
            result,
            Some(candidate_b),
            "MD5 tiebreaker should select the candidate with a matching chats directory"
        );
    }
}
