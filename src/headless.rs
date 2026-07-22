use crate::claude::{
    AgentContent, ContentBlock, LogEntry, UserContent, extract_tool_result_text,
    parse_agent_progress,
};
use crate::cli::{Command, DebugLevel, ListCommand, ProviderFilter, SearchCommand, ShowCommand};
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, ParseError, path_to_string, project_path_is_live,
};
use crate::providers::{LoadOptions, Provider};
use crate::tui::search;
use crate::tui::ui::extract_match_context;
use chrono::{DateTime, Duration, Local, NaiveDate, TimeZone};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

/// Width, in characters, of a search snippet window (roughly 240 chars centered
/// on a match). Also used as the minimum byte gap between two matches before a
/// second snippet is emitted, so windows do not overlap.
const SNIPPET_WIDTH: usize = 240;

/// Upper clamp on the number of snippets a single result may carry.
const SNIPPET_MAX: usize = 5;

pub struct HeadlessSettings {
    pub cli_local: bool,
    pub cli_global: bool,
    pub show_last: bool,
    pub show_deleted_projects: bool,
    pub debug: Option<DebugLevel>,
}

#[derive(Serialize)]
struct ConversationSummary {
    provider: &'static str,
    id: String,
    path: String,
    timestamp: String,
    project_name: Option<String>,
    project_path: Option<String>,
    cwd: Option<String>,
    preview: String,
    summary: Option<String>,
    model: Option<String>,
    message_count: usize,
    total_tokens: u64,
    duration_minutes: Option<u64>,
    parse_errors: Vec<ParseError>,
}

#[derive(Serialize)]
struct ConversationDetail {
    conversation: ConversationSummary,
    /// Total message count before `--grep` filtering. Present only when
    /// `--grep` is active, so plain `show` output is byte-for-byte unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    total_messages: Option<usize>,
    messages: Vec<MessageDto>,
}

#[derive(Serialize)]
struct SearchResult {
    provider: &'static str,
    id: String,
    path: String,
    timestamp: String,
    project_name: Option<String>,
    cwd: Option<String>,
    summary: Option<String>,
    score: f64,
    match_count: usize,
    snippets: Vec<String>,
}

#[derive(Serialize)]
struct MessageDto {
    index: usize,
    entry_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_index: Option<usize>,
    role: String,
    /// True on messages that matched a `--grep` pattern (neighbors kept for
    /// context are omitted). Absent without `--grep`, keeping output unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    matched: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_result_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_result_exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_result_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<Value>,
}

#[derive(Clone, Copy)]
struct MessageContext<'a> {
    timestamp: Option<&'a str>,
    model: Option<&'a str>,
    agent_id: Option<&'a str>,
    entry_index: usize,
    block_index: Option<usize>,
}

impl<'a> MessageContext<'a> {
    fn new(entry_index: usize) -> Self {
        Self {
            timestamp: None,
            model: None,
            agent_id: None,
            entry_index,
            block_index: None,
        }
    }

    fn timestamp(mut self, timestamp: &'a str) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    fn model(mut self, model: Option<&'a str>) -> Self {
        self.model = model;
        self
    }

    fn agent_id(mut self, agent_id: Option<&'a str>) -> Self {
        self.agent_id = agent_id;
        self
    }

    fn block_index(mut self, block_index: Option<usize>) -> Self {
        self.block_index = block_index;
        self
    }
}

impl MessageDto {
    fn new(role: impl Into<String>, context: MessageContext<'_>) -> Self {
        Self {
            index: 0,
            entry_index: context.entry_index,
            block_index: context.block_index,
            role: role.into(),
            matched: None,
            timestamp: context.timestamp.map(ToString::to_string),
            text: None,
            tool_call_id: None,
            tool_name: None,
            tool_input: None,
            tool_result: None,
            tool_result_status: None,
            tool_result_exit_code: None,
            tool_result_error: None,
            thinking: None,
            model: None,
            agent_id: None,
            subtype: None,
            level: None,
            duration_ms: None,
            source: None,
        }
    }
}

pub fn run_command(
    command: &Command,
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
) -> Result<()> {
    match command {
        Command::List(command) => run_list(command, providers, settings),
        Command::Show(command) => run_show(command, providers, settings),
        Command::Search(command) => run_search(command, providers, settings),
    }
}

fn run_list(
    command: &ListCommand,
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
) -> Result<()> {
    let mut conversations = load_conversations(
        providers,
        settings,
        command.provider,
        command.local,
        command.show_deleted_projects,
        command.cwd.is_some(),
        false,
    )?;
    apply_list_filters(&mut conversations, command)?;
    if let Some(limit) = command.limit {
        conversations.truncate(limit);
    }

    let summaries: Vec<_> = conversations
        .iter()
        .map(ConversationSummary::from_conversation)
        .collect();

    if command.jsonl {
        write_jsonl(&summaries)
    } else {
        write_json(&summaries)
    }
}

fn run_show(
    command: &ShowCommand,
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
) -> Result<()> {
    let conversations = load_conversations(
        providers,
        settings,
        command.provider,
        command.local,
        command.show_deleted_projects,
        false,
        false,
    )?;
    let conversation = resolve_conversation(&conversations, &command.target)?;
    let provider = providers
        .iter()
        .find(|provider| provider.kind() == conversation.provider)
        .ok_or_else(|| {
            AppError::CommandError(format!(
                "No provider found for {} conversation",
                conversation.provider.key()
            ))
        })?;
    let entries = provider.read_entries(conversation)?;
    let messages = messages_from_entries(&entries);

    // `--grep` keeps only messages matching a pattern (plus context neighbors)
    // and records the pre-filter count; without it the output is unchanged.
    let (messages, total_messages) = if command.grep.is_empty() {
        (messages, None)
    } else {
        let total = messages.len();
        (
            apply_grep(messages, &command.grep, command.context),
            Some(total),
        )
    };

    let detail = ConversationDetail {
        conversation: ConversationSummary::from_conversation(conversation),
        total_messages,
        messages,
    };

    write_json(&detail)
}

/// Keep only messages matching any `--grep` pattern (case-insensitive substring,
/// OR across patterns), plus up to `context` neighbors on each side. Matched
/// messages are flagged with `matched = Some(true)`; context-only neighbors are
/// not. Overlapping/adjacent context ranges are merged.
fn apply_grep(messages: Vec<MessageDto>, patterns: &[String], context: usize) -> Vec<MessageDto> {
    let patterns_lower: Vec<String> = patterns.iter().map(|p| p.to_lowercase()).collect();

    let matched: Vec<bool> = messages
        .iter()
        .map(|message| message_matches_grep(message, &patterns_lower))
        .collect();

    // A boolean mask of which message indices survive: each match contributes
    // the inclusive window [i - context, i + context], clamped to the ends.
    let mut keep = vec![false; messages.len()];
    for (i, is_match) in matched.iter().enumerate() {
        if *is_match {
            let lo = i.saturating_sub(context);
            let hi = i
                .saturating_add(context)
                .min(messages.len().saturating_sub(1));
            for slot in &mut keep[lo..=hi] {
                *slot = true;
            }
        }
    }

    messages
        .into_iter()
        .zip(matched)
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, (mut message, is_match))| {
            if is_match {
                message.matched = Some(true);
            }
            message
        })
        .collect()
}

/// Whether a message's text, thinking, or stringified tool input/result contains
/// any of the already-lowercased patterns. The JSON stringification happens only
/// here (i.e. only when `--grep` is active).
fn message_matches_grep(message: &MessageDto, patterns_lower: &[String]) -> bool {
    let mut haystack = String::new();
    if let Some(text) = &message.text {
        haystack.push_str(text);
        haystack.push('\n');
    }
    if let Some(thinking) = &message.thinking {
        haystack.push_str(thinking);
        haystack.push('\n');
    }
    if let Some(input) = &message.tool_input
        && let Ok(rendered) = serde_json::to_string(input)
    {
        haystack.push_str(&rendered);
        haystack.push('\n');
    }
    if let Some(result) = &message.tool_result
        && let Ok(rendered) = serde_json::to_string(result)
    {
        haystack.push_str(&rendered);
        haystack.push('\n');
    }

    let haystack_lower = haystack.to_lowercase();
    patterns_lower
        .iter()
        .any(|pattern| haystack_lower.contains(pattern))
}

fn run_search(
    command: &SearchCommand,
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
) -> Result<()> {
    let mut conversations = load_conversations(
        providers,
        settings,
        command.provider,
        command.local,
        false,
        command.cwd.is_some(),
        true,
    )?;
    apply_search_filters(&mut conversations, command)?;

    let results = rank_and_build(conversations, command, Local::now());

    if command.jsonl {
        write_jsonl(&results)
    } else {
        write_json(&results)
    }
}

/// Rank the (already time/provider/scope-filtered) conversations against the
/// query, drop excluded sessions, and build the output rows (snippets and
/// match counts) for the top `limit`.
///
/// Ranking, snippet extraction, and match counting all reuse the TUI search
/// machinery so headless order matches the interactive list exactly.
fn rank_and_build(
    mut conversations: Vec<Conversation>,
    command: &SearchCommand,
    now: DateTime<Local>,
) -> Vec<SearchResult> {
    if !command.exclude_session.is_empty() {
        let excluded: HashSet<&str> = command.exclude_session.iter().map(String::as_str).collect();
        conversations.retain(|conversation| !excluded.contains(conversation.id.as_str()));
    }

    let query = command.words.join(" ");
    let snippet_count = command.snippets.min(SNIPPET_MAX);

    // `precompute_search_text` moves each conversation's body into `text_lower`,
    // so keep the searchable rows alive for snippet slicing below.
    let searchable = search::precompute_search_text(&mut conversations);
    let mut text_lowers: Vec<&str> = vec![""; conversations.len()];
    for row in &searchable {
        if row.index < text_lowers.len() {
            text_lowers[row.index] = &row.text_lower;
        }
    }

    let ranked = search::search_scored(&conversations, &searchable, &query, now, None);

    let mut results = Vec::new();
    for (index, score) in ranked {
        if results.len() >= command.limit {
            break;
        }
        let conversation = &conversations[index];
        let text_lower = text_lowers[index];
        let offsets = search::match_offsets(text_lower, &query);
        let snippets = build_snippets(text_lower, &offsets, snippet_count);
        results.push(SearchResult {
            provider: conversation.provider.key(),
            id: conversation.id.clone(),
            path: conversation.path.to_string_lossy().to_string(),
            timestamp: conversation.timestamp.to_rfc3339(),
            project_name: conversation.project_name.clone(),
            cwd: path_to_string(conversation.cwd.as_deref()),
            summary: conversation.summary.clone(),
            score,
            match_count: offsets.len(),
            snippets,
        });
    }

    results
}

/// Slice up to `count` ~240-char snippet windows from the lowercased corpus,
/// centered on the earliest match offsets. Offsets closer than one window width
/// to the previous snippet are skipped so windows don't overlap.
fn build_snippets(text_lower: &str, offsets: &[(usize, usize)], count: usize) -> Vec<String> {
    if count == 0 {
        return Vec::new();
    }

    let mut snippets = Vec::new();
    let mut last_pos: Option<usize> = None;
    for &(pos, char_len) in offsets {
        if snippets.len() >= count {
            break;
        }
        if let Some(previous) = last_pos
            && pos < previous + SNIPPET_WIDTH
        {
            continue;
        }
        snippets.push(extract_match_context(
            text_lower,
            pos,
            char_len,
            SNIPPET_WIDTH,
        ));
        last_pos = Some(pos);
    }
    snippets
}

fn apply_search_filters(
    conversations: &mut Vec<Conversation>,
    command: &SearchCommand,
) -> Result<()> {
    let after = after_cutoff(command.since.as_deref(), command.after.as_deref())?;
    let before = command
        .before
        .as_deref()
        .map(|value| parse_timestamp_filter(value, "--before"))
        .transpose()?;
    let cwd_roots = command
        .cwd
        .as_deref()
        .map(crate::loader::filter_path_roots)
        .transpose()?;

    conversations.retain(|conversation| {
        after.is_none_or(|after| conversation.timestamp >= after)
            && before.is_none_or(|before| conversation.timestamp < before)
            && cwd_roots
                .as_ref()
                .is_none_or(|roots| crate::loader::conversation_matches_scope(conversation, roots))
    });

    Ok(())
}

fn load_conversations(
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
    provider_filter: Option<ProviderFilter>,
    command_local: bool,
    command_show_deleted: bool,
    force_global: bool,
    include_full_text: bool,
) -> Result<Vec<Conversation>> {
    let local = use_local_scope(settings, command_local, force_global);
    let show_deleted_projects = command_show_deleted || settings.show_deleted_projects;
    // Headless `list`/`show` never emit the conversation body, so they load under
    // the metadata profile (`include_full_text = false`): the corpus is never
    // decoded from the cache, Cursor skips deriving it, and the returned
    // conversations carry an empty body. A file provider's fresh parse still
    // computes the text as a scan byproduct and writes the cache row complete, so
    // a later full-profile run that hits those rows gets a real search corpus.
    // `search` needs the body to rank and snippet, so it passes `true`.
    let load_options = LoadOptions {
        show_last: settings.show_last,
        debug: settings.debug,
        include_full_text,
    };
    let mut conversations = if local {
        crate::loader::load_local(providers, load_options, provider_filter)?
    } else {
        load_global_conversations(providers, load_options, provider_filter)?
    };

    if !show_deleted_projects {
        conversations.retain(|conversation| {
            conversation
                .project_path
                .as_ref()
                .is_none_or(|path| project_path_is_live(path))
        });
    }

    conversations.sort_by_key(|c| std::cmp::Reverse(c.timestamp));

    Ok(conversations)
}

fn use_local_scope(settings: &HeadlessSettings, command_local: bool, force_global: bool) -> bool {
    // Headless commands default to global output even when the interactive TUI
    // config has `local = true`; scripts and skills need stable scope unless the
    // caller explicitly passes --local.
    //
    // An explicit `--global` (cli_global) — like `--cwd`'s force_global, which
    // loads everything before filtering by path — overrides any --local request,
    // so `mnemonai --global list --local` returns global results as the flag promises.
    !force_global && !settings.cli_global && (command_local || settings.cli_local)
}

fn apply_list_filters(conversations: &mut Vec<Conversation>, command: &ListCommand) -> Result<()> {
    let after = list_after_cutoff(command)?;
    let before = command
        .before
        .as_deref()
        .map(|value| parse_timestamp_filter(value, "--before"))
        .transpose()?;
    let cwd_roots = command
        .cwd
        .as_deref()
        .map(crate::loader::filter_path_roots)
        .transpose()?;

    conversations.retain(|conversation| {
        after.is_none_or(|after| conversation.timestamp >= after)
            && before.is_none_or(|before| conversation.timestamp < before)
            && cwd_roots
                .as_ref()
                .is_none_or(|roots| crate::loader::conversation_matches_scope(conversation, roots))
    });

    Ok(())
}

fn list_after_cutoff(command: &ListCommand) -> Result<Option<DateTime<Local>>> {
    after_cutoff(command.since.as_deref(), command.after.as_deref())
}

/// Resolve the inclusive lower time bound from `--since` (a relative duration) or
/// `--after` (an absolute timestamp). Shared by `list` and `search`.
fn after_cutoff(since: Option<&str>, after: Option<&str>) -> Result<Option<DateTime<Local>>> {
    if let Some(since) = since {
        let duration = parse_since_duration(since)?;
        let cutoff = Local::now()
            .checked_sub_signed(duration)
            .ok_or_else(|| invalid_duration(since))?;
        Ok(Some(cutoff))
    } else {
        after
            .map(|value| parse_timestamp_filter(value, "--after"))
            .transpose()
    }
}

fn parse_since_duration(value: &str) -> Result<Duration> {
    let trimmed = value.trim();
    let unit_start = trimmed
        .find(|ch: char| !ch.is_ascii_digit())
        .ok_or_else(|| invalid_duration(value))?;
    let (amount, unit) = trimmed.split_at(unit_start);
    if amount.is_empty() || unit.is_empty() || unit.chars().any(char::is_whitespace) {
        return Err(invalid_duration(value));
    }

    let amount = amount.parse::<i64>().map_err(|_| invalid_duration(value))?;
    if amount <= 0 {
        return Err(invalid_duration(value));
    }

    // Use the checked constructors: an out-of-range amount (e.g. 9999999999999w)
    // would otherwise panic inside chrono rather than return a clean error.
    let duration = match unit.to_ascii_lowercase().as_str() {
        "m" | "min" | "mins" | "minute" | "minutes" => Duration::try_minutes(amount),
        "h" | "hr" | "hrs" | "hour" | "hours" => Duration::try_hours(amount),
        "d" | "day" | "days" => Duration::try_days(amount),
        "w" | "week" | "weeks" => Duration::try_weeks(amount),
        _ => return Err(invalid_duration(value)),
    };
    duration.ok_or_else(|| invalid_duration(value))
}

fn invalid_duration(value: &str) -> AppError {
    AppError::CommandError(format!(
        "Invalid --since value '{}'; expected a positive duration like 7d, 24h, or 2w",
        value
    ))
}

fn parse_timestamp_filter(value: &str, flag: &str) -> Result<DateTime<Local>> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(timestamp.with_timezone(&Local));
    }

    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        let midnight = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| invalid_timestamp(value, flag))?;
        // `.single()` is None when local midnight falls in a DST spring-forward
        // gap; fall back to the earliest valid instant so a real date is never
        // rejected purely because of a timezone transition.
        return Local
            .from_local_datetime(&midnight)
            .earliest()
            .ok_or_else(|| invalid_timestamp(value, flag));
    }

    Err(invalid_timestamp(value, flag))
}

fn invalid_timestamp(value: &str, flag: &str) -> AppError {
    AppError::CommandError(format!(
        "Invalid {} value '{}'; expected RFC 3339 or YYYY-MM-DD",
        flag, value
    ))
}

fn load_global_conversations(
    providers: &[Box<dyn Provider>],
    options: LoadOptions,
    provider_filter: Option<ProviderFilter>,
) -> Result<Vec<Conversation>> {
    // Start every matching provider's loader before draining any receiver so
    // the providers load in parallel: total latency is the slowest provider
    // rather than the sum of all of them. Channels buffer, so draining them
    // one at a time in provider order afterwards preserves the ordering and
    // per-provider semantics of the previous sequential loop.
    let receivers: Vec<_> = providers
        .iter()
        .filter(|provider| {
            crate::loader::provider_filter_matches(provider_filter, &provider.kind())
        })
        .map(|provider| provider.load_conversations_streaming(options))
        .collect();

    let mut conversations = Vec::new();
    for rx in receivers {
        for message in rx {
            match message {
                LoaderMessage::Batch(mut batch) => conversations.append(&mut batch),
                LoaderMessage::Done => break,
                LoaderMessage::ProjectError => {}
                LoaderMessage::Fatal(error) => {
                    if provider_filter.is_some() {
                        return Err(error);
                    }
                    break;
                }
            }
        }
    }

    Ok(conversations)
}

fn resolve_conversation<'a>(
    conversations: &'a [Conversation],
    target: &str,
) -> Result<&'a Conversation> {
    let target_path = Path::new(target);
    // Canonicalized once up front; the per-conversation canonicalization in
    // `path_matches` is gated behind cheap comparisons so resolving a target
    // does not pay one realpath syscall per loaded conversation.
    let target_canonical = target_path.canonicalize().ok();

    let matches: Vec<&Conversation> = conversations
        .iter()
        .filter(|conversation| {
            conversation.id == target
                || conversation
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem == target)
                || path_matches(&conversation.path, target_path, target_canonical.as_deref())
        })
        .collect();

    match matches.as_slice() {
        [conversation] => Ok(conversation),
        [] => Err(AppError::CommandError(format!(
            "No conversation found for target '{}'",
            target
        ))),
        matches => {
            let rendered = matches
                .iter()
                .take(8)
                .map(|conversation| {
                    format!(
                        "{}:{} ({})",
                        conversation.provider.key(),
                        conversation.id,
                        conversation.path.display()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(AppError::CommandError(format!(
                "Ambiguous conversation target '{}'; matches: {}. \
                 Pass --provider <name> to disambiguate.",
                target, rendered
            )))
        }
    }
}

fn path_matches(path: &Path, target: &Path, target_canonical: Option<&Path>) -> bool {
    if path == target {
        return true;
    }
    let Some(canonical) = target_canonical else {
        return false;
    };

    // Canonicalizing costs a realpath syscall per conversation, so only pay it
    // when a file-name prefilter passes. `canonicalize` preserves the final
    // component's name unless that component is itself a symlink, so a
    // canonical match implies the conversation's file name equals either the
    // canonical target's name (regular session file) or the literal target's
    // name (conversation path and target name the same symlink).
    let name = path.file_name();
    if name != canonical.file_name() && name != target.file_name() {
        return false;
    }

    path.canonicalize()
        .map(|path| path == canonical)
        .unwrap_or(false)
}

impl ConversationSummary {
    fn from_conversation(conversation: &Conversation) -> Self {
        Self {
            provider: conversation.provider.key(),
            id: conversation.id.clone(),
            path: conversation.path.to_string_lossy().to_string(),
            timestamp: conversation.timestamp.to_rfc3339(),
            project_name: conversation.project_name.clone(),
            project_path: path_to_string(conversation.project_path.as_deref()),
            cwd: path_to_string(conversation.cwd.as_deref()),
            preview: conversation.preview.clone(),
            summary: conversation.summary.clone(),
            model: conversation.model.clone(),
            message_count: conversation.message_count,
            total_tokens: conversation.total_tokens,
            duration_minutes: conversation.duration_minutes,
            parse_errors: conversation.parse_errors.clone(),
        }
    }
}

fn messages_from_entries(entries: &[LogEntry]) -> Vec<MessageDto> {
    let mut messages = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        match entry {
            LogEntry::Summary { summary } => {
                let mut message = MessageDto::new("summary", MessageContext::new(entry_index));
                message.text = Some(summary.clone());
                push_message(&mut messages, message);
            }
            LogEntry::User {
                message, timestamp, ..
            } => push_user_message(&mut messages, message, timestamp, entry_index),
            LogEntry::Assistant {
                message, timestamp, ..
            } => push_assistant_message(&mut messages, message, timestamp, entry_index),
            LogEntry::Progress { data, .. } => {
                if let Some(progress) = parse_agent_progress(data) {
                    let AgentContent::Blocks(blocks) = &progress.message.message.content;
                    let role = format!("agent_{}", progress.message.message_type);
                    for (block_index, block) in blocks.iter().enumerate() {
                        let context = MessageContext::new(entry_index)
                            .agent_id(Some(progress.agent_id.as_str()))
                            .block_index(Some(block_index));
                        push_content_block(&mut messages, &role, block, context);
                    }
                }
            }
            LogEntry::System {
                subtype,
                level,
                duration_ms,
                ..
            } => {
                let mut message = MessageDto::new("system", MessageContext::new(entry_index));
                message.subtype = Some(subtype.clone());
                // `level` is a log severity ("info"/"warning"/...), not chat
                // content, so keep it out of the `text` field consumers read.
                message.level = level.clone();
                message.duration_ms = *duration_ms;
                push_message(&mut messages, message);
            }
            LogEntry::FileHistorySnapshot | LogEntry::Unknown => {}
        }
    }
    messages
}

fn push_message(messages: &mut Vec<MessageDto>, mut message: MessageDto) {
    message.index = messages.len();
    messages.push(message);
}

fn push_user_message(
    messages: &mut Vec<MessageDto>,
    message: &crate::claude::UserMessage,
    timestamp: &str,
    entry_index: usize,
) {
    match &message.content {
        UserContent::String(text) => {
            let context = MessageContext::new(entry_index).timestamp(timestamp);
            push_text_message(messages, "user", text, context);
        }
        UserContent::Blocks(blocks) => {
            for (block_index, block) in blocks.iter().enumerate() {
                let context = MessageContext::new(entry_index)
                    .timestamp(timestamp)
                    .block_index(Some(block_index));
                push_content_block(messages, "user", block, context);
            }
        }
    }
}

fn push_assistant_message(
    messages: &mut Vec<MessageDto>,
    message: &crate::claude::AssistantMessage,
    timestamp: &str,
    entry_index: usize,
) {
    for (block_index, block) in message.content.iter().enumerate() {
        let context = MessageContext::new(entry_index)
            .timestamp(timestamp)
            .model(message.model.as_deref())
            .block_index(Some(block_index));
        push_content_block(messages, "assistant", block, context);
    }
}

fn push_content_block(
    messages: &mut Vec<MessageDto>,
    text_role: &str,
    block: &ContentBlock,
    context: MessageContext<'_>,
) {
    match block {
        ContentBlock::Text { text } => {
            push_text_message(messages, text_role, text, context);
        }
        ContentBlock::ToolUse { id, name, input } => {
            let mut message = MessageDto::new("tool_call", context);
            message.tool_call_id = Some(id.clone());
            message.tool_name = Some(name.clone());
            message.tool_input = Some(input.clone());
            message.model = context.model.map(ToString::to_string);
            message.agent_id = context.agent_id.map(ToString::to_string);
            push_message(messages, message);
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            status,
        } => {
            let mut message = MessageDto::new("tool_result", context);
            message.tool_call_id = Some(tool_use_id.clone());
            message.text = extract_tool_result_text(content.as_ref());
            // Omit an empty result body rather than emitting `"tool_result": null`
            // (some providers default missing content to JSON null).
            message.tool_result = content.clone().filter(|value| !value.is_null());
            message.tool_result_status = status.clone();
            message.tool_result_exit_code = message.text.as_deref().and_then(tool_result_exit_code);
            message.tool_result_error = merge_error_markers(&[
                *is_error,
                message.tool_result_exit_code.map(|code| code != 0),
                status.as_deref().and_then(status_is_error),
            ]);
            message.agent_id = context.agent_id.map(ToString::to_string);
            push_message(messages, message);
        }
        ContentBlock::Thinking { thinking, .. } => {
            let mut message = MessageDto::new("thinking", context);
            message.thinking = Some(thinking.clone());
            message.model = context.model.map(ToString::to_string);
            message.agent_id = context.agent_id.map(ToString::to_string);
            push_message(messages, message);
        }
        ContentBlock::Image { source } => {
            let mut message = MessageDto::new("image", context);
            message.source = Some(source.clone());
            message.agent_id = context.agent_id.map(ToString::to_string);
            push_message(messages, message);
        }
    }
}

fn push_text_message(
    messages: &mut Vec<MessageDto>,
    role: &str,
    text: &str,
    context: MessageContext<'_>,
) {
    if text.trim().is_empty() {
        return;
    }

    let mut message = MessageDto::new(role, context);
    message.text = Some(text.to_string());
    message.model = context.model.map(ToString::to_string);
    message.agent_id = context.agent_id.map(ToString::to_string);
    push_message(messages, message);
}

fn status_is_error(status: &str) -> Option<bool> {
    match status.to_ascii_lowercase().as_str() {
        "error" | "errored" | "failed" | "failure" | "cancelled" | "canceled" => Some(true),
        "success" | "succeeded" | "complete" | "completed" | "ok" | "done" => Some(false),
        other if other.contains("error") || other.contains("fail") => Some(true),
        _ => None,
    }
}

fn tool_result_exit_code(text: &str) -> Option<i32> {
    const MARKERS: &[&str] = &["Process exited with code ", "Exit code "];

    text.lines().take(8).find_map(|line| {
        let line = line.trim();
        MARKERS.iter().find_map(|marker| {
            let rest = line.strip_prefix(marker)?;
            // Require the marker line to be exactly "<marker><integer>" with no
            // trailing text, so prose like "Exit code 2 indicates a syntax error"
            // is not misread as a real exit code.
            let mut tokens = rest.split_whitespace();
            let code = tokens.next()?.parse().ok()?;
            tokens.next().is_none().then_some(code)
        })
    })
}

fn merge_error_markers(markers: &[Option<bool>]) -> Option<bool> {
    if markers.contains(&Some(true)) {
        Some(true)
    } else if markers.iter().any(Option::is_some) {
        Some(false)
    } else {
        None
    }
}

fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut buf = serde_json::to_vec_pretty(value)?;
    buf.push(b'\n');
    Ok(buf)
}

/// Serialize each value as a compact JSON object on its own line (JSONL).
fn jsonl_bytes<T: Serialize>(values: &[T]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    for value in values {
        serde_json::to_writer(&mut buf, value)?;
        buf.push(b'\n');
    }
    Ok(buf)
}

fn write_json<T: Serialize>(value: &T) -> Result<()> {
    write_stdout(&json_bytes(value)?)
}

fn write_jsonl<T: Serialize>(values: &[T]) -> Result<()> {
    write_stdout(&jsonl_bytes(values)?)
}

/// Write a fully-serialized buffer to stdout in a single call.
///
/// Serializing into memory first means a closed downstream pipe (e.g.
/// `mnemonai list | head`) surfaces as a plain `io::ErrorKind::BrokenPipe` from
/// this one `write_all` rather than a misleading "JSON parsing error", letting
/// `main` exit cleanly.
fn write_stdout(buf: &[u8]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(buf)?;
    handle.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude::{AssistantMessage, TokenUsage, UserMessage};
    use crate::history::ProviderKind;
    use chrono::{Local, TimeZone};
    use std::path::PathBuf;

    fn conversation(id: &str, path: &str, provider: ProviderKind) -> Conversation {
        Conversation {
            path: PathBuf::from(path),
            provider,
            id: id.to_string(),
            timestamp: Local.timestamp_opt(1_700_000_000, 0).single().unwrap(),
            preview: "preview".to_string(),
            full_text: "full text".to_string(),
            project_name: Some("project".to_string()),
            project_path: Some(PathBuf::from("/tmp/project")),
            cwd: Some(PathBuf::from("/tmp/project")),
            message_count: 2,
            parse_errors: Vec::new(),
            summary: Some("summary".to_string()),
            model: Some("model".to_string()),
            total_tokens: 42,
            duration_minutes: Some(3),
        }
    }

    struct StubProvider {
        kind: ProviderKind,
        conversations: Vec<Conversation>,
        fatal: bool,
    }

    impl StubProvider {
        fn ok(kind: ProviderKind, conversations: Vec<Conversation>) -> Box<dyn Provider> {
            Box::new(Self {
                kind,
                conversations,
                fatal: false,
            })
        }

        fn fatal(kind: ProviderKind) -> Box<dyn Provider> {
            Box::new(Self {
                kind,
                conversations: Vec::new(),
                fatal: true,
            })
        }
    }

    impl Provider for StubProvider {
        fn kind(&self) -> ProviderKind {
            self.kind
        }

        fn name(&self) -> &str {
            "stub"
        }

        fn load_conversations(&self, _options: LoadOptions) -> Result<Vec<Conversation>> {
            if self.fatal {
                Err(AppError::CommandError("stub failure".to_string()))
            } else {
                Ok(self.conversations.clone())
            }
        }

        fn load_conversations_streaming(
            &self,
            _options: LoadOptions,
        ) -> std::sync::mpsc::Receiver<LoaderMessage> {
            let (tx, rx) = std::sync::mpsc::channel();
            if self.fatal {
                let _ = tx.send(LoaderMessage::Fatal(AppError::CommandError(
                    "stub failure".to_string(),
                )));
            } else {
                let _ = tx.send(LoaderMessage::Batch(self.conversations.clone()));
                let _ = tx.send(LoaderMessage::Done);
            }
            rx
        }

        fn read_entries(&self, _conversation: &Conversation) -> Result<Vec<LogEntry>> {
            Ok(Vec::new())
        }

        fn resume(&self, _conversation: &Conversation, _default_args: &[String]) -> Result<()> {
            Ok(())
        }

        fn delete(&self, _conversation: &Conversation) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn summary_uses_stable_provider_key() {
        let conversation = conversation("abc", "/tmp/abc.jsonl", ProviderKind::CursorAgent);
        let summary = ConversationSummary::from_conversation(&conversation);
        let json = serde_json::to_value(summary).unwrap();

        assert_eq!(json["provider"], "cursor-agent");
        assert_eq!(json["id"], "abc");
        assert_eq!(json["path"], "/tmp/abc.jsonl");
    }

    #[test]
    fn converts_entries_to_structured_messages() {
        let entries = vec![
            LogEntry::User {
                message: UserMessage {
                    content: UserContent::String("run tests".to_string()),
                },
                timestamp: "2026-06-19T10:00:00-07:00".to_string(),
                cwd: None,
            },
            LogEntry::Assistant {
                message: AssistantMessage {
                    content: vec![
                        ContentBlock::Text {
                            text: "I'll run them.".to_string(),
                        },
                        ContentBlock::ToolUse {
                            id: "tool-1".to_string(),
                            name: "Bash".to_string(),
                            input: serde_json::json!({"cmd": "cargo test"}),
                        },
                        ContentBlock::Thinking {
                            thinking: "Need to verify failures.".to_string(),
                        },
                    ],
                    model: Some("claude-test".to_string()),
                    usage: Some(TokenUsage::default()),
                    id: None,
                },
                timestamp: "2026-06-19T10:00:01-07:00".to_string(),
            },
        ];

        let messages = messages_from_entries(&entries);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].index, 0);
        assert_eq!(messages[0].entry_index, 0);
        assert_eq!(messages[0].block_index, None);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].entry_index, 1);
        assert_eq!(messages[1].block_index, Some(0));
        assert_eq!(messages[2].role, "tool_call");
        assert_eq!(messages[2].entry_index, 1);
        assert_eq!(messages[2].block_index, Some(1));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("tool-1"));
        assert_eq!(messages[2].tool_name.as_deref(), Some("Bash"));
        assert_eq!(messages[3].role, "thinking");
        assert_eq!(messages[3].model.as_deref(), Some("claude-test"));
    }

    #[test]
    fn resolves_unique_id() {
        let conversations = vec![
            conversation("abc", "/tmp/abc.jsonl", ProviderKind::Claude),
            conversation("def", "/tmp/def.jsonl", ProviderKind::Codex),
        ];

        let resolved = resolve_conversation(&conversations, "def").unwrap();

        assert_eq!(resolved.provider, ProviderKind::Codex);
    }

    #[test]
    fn rejects_ambiguous_id() {
        let conversations = vec![
            conversation("same", "/tmp/a.jsonl", ProviderKind::Claude),
            conversation("same", "/tmp/b.jsonl", ProviderKind::Codex),
        ];

        let err = match resolve_conversation(&conversations, "same") {
            Ok(_) => panic!("expected ambiguous conversation error"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("Ambiguous conversation target"));
    }

    #[test]
    fn ambiguity_error_suggests_provider() {
        let conversations = vec![
            conversation("same", "/tmp/a.jsonl", ProviderKind::Claude),
            conversation("same", "/tmp/b.jsonl", ProviderKind::Codex),
        ];

        let err = match resolve_conversation(&conversations, "same") {
            Ok(_) => panic!("expected ambiguous conversation error"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("--provider"));
    }

    #[test]
    fn resolves_by_path_and_file_stem() {
        let conversations = vec![
            conversation("id-a", "/tmp/sess-a.jsonl", ProviderKind::Claude),
            conversation("id-b", "/tmp/sess-b.jsonl", ProviderKind::Codex),
        ];

        let by_path = resolve_conversation(&conversations, "/tmp/sess-b.jsonl").unwrap();
        assert_eq!(by_path.id, "id-b");

        let by_stem = resolve_conversation(&conversations, "sess-a").unwrap();
        assert_eq!(by_stem.id, "id-a");
    }

    #[test]
    fn rejects_unknown_target() {
        let conversations = vec![conversation(
            "id-a",
            "/tmp/sess-a.jsonl",
            ProviderKind::Claude,
        )];

        let err = match resolve_conversation(&conversations, "does-not-exist") {
            Ok(_) => panic!("expected no-conversation error"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("No conversation found"));
    }

    #[test]
    fn resolves_path_forms_through_canonicalization() {
        // std::env::temp_dir() sits behind a symlink on macOS (/var ->
        // /private/var), so the literal and canonical forms differ there; on
        // platforms where they coincide this degenerates to literal equality.
        let dir =
            std::env::temp_dir().join(format!("mnemonai-headless-resolve-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let literal = dir.join("sess-canon.jsonl");
        std::fs::write(&literal, "{}\n").unwrap();
        let canonical = literal.canonicalize().unwrap();

        // Stored canonical, target literal.
        let conversations = vec![
            conversation(
                "id-canon",
                canonical.to_str().unwrap(),
                ProviderKind::Claude,
            ),
            conversation("id-other", "/tmp/other.jsonl", ProviderKind::Codex),
        ];
        let resolved = resolve_conversation(&conversations, literal.to_str().unwrap()).unwrap();
        assert_eq!(resolved.id, "id-canon");

        // Stored literal, target canonical.
        let conversations = vec![conversation(
            "id-literal",
            literal.to_str().unwrap(),
            ProviderKind::Claude,
        )];
        let resolved = resolve_conversation(&conversations, canonical.to_str().unwrap()).unwrap();
        assert_eq!(resolved.id, "id-literal");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn resolves_symlinked_target_to_stored_real_path() {
        // A target that is a symlink to the stored session file must still
        // resolve even though the two paths share no literal component; the
        // file-name prefilter must compare against the canonical target name.
        let dir =
            std::env::temp_dir().join(format!("mnemonai-headless-symlink-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real-session.jsonl");
        std::fs::write(&real, "{}\n").unwrap();
        let link = dir.join("link-session.jsonl");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let stored = real.canonicalize().unwrap();
        let conversations = vec![conversation(
            "id-real",
            stored.to_str().unwrap(),
            ProviderKind::Claude,
        )];

        let resolved = resolve_conversation(&conversations, link.to_str().unwrap()).unwrap();
        assert_eq!(resolved.id, "id-real");

        // Stored symlink path, target the same symlink through the canonical
        // directory form: names match the literal target, not the canonical one.
        let conversations = vec![conversation(
            "id-link",
            link.to_str().unwrap(),
            ProviderKind::Claude,
        )];
        let target = dir.canonicalize().unwrap().join("link-session.jsonl");
        let resolved = resolve_conversation(&conversations, target.to_str().unwrap()).unwrap();
        assert_eq!(resolved.id, "id-link");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn global_load_appends_providers_in_provider_order() {
        let providers: Vec<Box<dyn Provider>> = vec![
            StubProvider::ok(
                ProviderKind::Claude,
                vec![conversation(
                    "claude-1",
                    "/tmp/claude-1.jsonl",
                    ProviderKind::Claude,
                )],
            ),
            StubProvider::ok(
                ProviderKind::Codex,
                vec![conversation(
                    "codex-1",
                    "/tmp/codex-1.jsonl",
                    ProviderKind::Codex,
                )],
            ),
        ];

        let conversations = load_global_conversations(
            &providers,
            LoadOptions {
                show_last: false,
                debug: None,
                include_full_text: false,
            },
            None,
        )
        .unwrap();

        let ids: Vec<_> = conversations.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["claude-1", "codex-1"]);
    }

    #[test]
    fn global_load_skips_fatal_provider_without_filter() {
        let providers: Vec<Box<dyn Provider>> = vec![
            StubProvider::fatal(ProviderKind::Claude),
            StubProvider::ok(
                ProviderKind::Codex,
                vec![conversation(
                    "codex-1",
                    "/tmp/codex-1.jsonl",
                    ProviderKind::Codex,
                )],
            ),
        ];

        let conversations = load_global_conversations(
            &providers,
            LoadOptions {
                show_last: false,
                debug: None,
                include_full_text: false,
            },
            None,
        )
        .unwrap();

        let ids: Vec<_> = conversations.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["codex-1"]);
    }

    #[test]
    fn global_load_fatal_is_an_error_with_provider_filter() {
        let providers: Vec<Box<dyn Provider>> = vec![StubProvider::fatal(ProviderKind::Claude)];

        let result = load_global_conversations(
            &providers,
            LoadOptions {
                show_last: false,
                debug: None,
                include_full_text: false,
            },
            Some(ProviderFilter::Claude),
        );

        assert!(result.is_err());
    }

    #[test]
    fn global_load_honors_provider_filter() {
        let providers: Vec<Box<dyn Provider>> = vec![
            StubProvider::ok(
                ProviderKind::Claude,
                vec![conversation(
                    "claude-1",
                    "/tmp/claude-1.jsonl",
                    ProviderKind::Claude,
                )],
            ),
            StubProvider::ok(
                ProviderKind::Codex,
                vec![conversation(
                    "codex-1",
                    "/tmp/codex-1.jsonl",
                    ProviderKind::Codex,
                )],
            ),
        ];

        let conversations = load_global_conversations(
            &providers,
            LoadOptions {
                show_last: false,
                debug: None,
                include_full_text: false,
            },
            Some(ProviderFilter::Codex),
        )
        .unwrap();

        let ids: Vec<_> = conversations.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["codex-1"]);
    }

    #[test]
    fn normalizes_system_image_tool_result_and_summary_entries() {
        let entries = vec![
            LogEntry::Summary {
                summary: "Session title".to_string(),
            },
            // Blank user text is dropped rather than emitted as an empty message.
            LogEntry::User {
                message: UserMessage {
                    content: UserContent::String("   ".to_string()),
                },
                timestamp: "2026-06-19T10:00:00-07:00".to_string(),
                cwd: None,
            },
            LogEntry::Assistant {
                message: AssistantMessage {
                    content: vec![
                        ContentBlock::ToolResult {
                            tool_use_id: "t1".to_string(),
                            content: Some(serde_json::json!([
                                {"type": "text", "text": "line one"},
                                {"type": "text", "text": "line two"},
                            ])),
                            is_error: Some(true),
                            status: Some("failed".to_string()),
                        },
                        ContentBlock::Image {
                            source: serde_json::json!({"type": "base64", "media_type": "image/png"}),
                        },
                    ],
                    model: Some("claude-test".to_string()),
                    usage: None,
                    id: None,
                },
                timestamp: "2026-06-19T10:00:01-07:00".to_string(),
            },
            LogEntry::System {
                subtype: "turn_duration".to_string(),
                level: Some("warning".to_string()),
                duration_ms: Some(1234),
            },
        ];

        let messages = messages_from_entries(&entries);

        let roles: Vec<_> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["summary", "tool_result", "image", "system"]);

        assert_eq!(messages[0].text.as_deref(), Some("Session title"));

        assert_eq!(messages[1].text.as_deref(), Some("line one\n\nline two"));
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(messages[1].tool_result_status.as_deref(), Some("failed"));
        assert_eq!(messages[1].tool_result_error, Some(true));
        assert!(messages[1].tool_result.is_some());

        assert!(messages[2].source.is_some());

        let system = &messages[3];
        assert_eq!(system.subtype.as_deref(), Some("turn_duration"));
        assert_eq!(system.level.as_deref(), Some("warning"));
        assert_eq!(system.duration_ms, Some(1234));
        assert!(
            system.text.is_none(),
            "system log level must not leak into the text field"
        );
    }

    #[test]
    fn jsonl_bytes_emit_one_compact_object_per_line() {
        let summaries = vec![
            ConversationSummary::from_conversation(&conversation(
                "a",
                "/tmp/a.jsonl",
                ProviderKind::Claude,
            )),
            ConversationSummary::from_conversation(&conversation(
                "b",
                "/tmp/b.jsonl",
                ProviderKind::Codex,
            )),
        ];

        let text = String::from_utf8(jsonl_bytes(&summaries).unwrap()).unwrap();
        let lines: Vec<_> = text.lines().collect();

        assert_eq!(lines.len(), 2);
        assert!(
            !text.trim_start().starts_with('['),
            "JSONL must not be an array"
        );
        let first: Value = serde_json::from_str(lines[0]).unwrap();
        let second: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(first["id"], "a");
        assert_eq!(second["id"], "b");
    }

    #[test]
    fn parses_since_duration_values() {
        assert_eq!(parse_since_duration("15m").unwrap(), Duration::minutes(15));
        assert_eq!(parse_since_duration("24h").unwrap(), Duration::hours(24));
        assert_eq!(parse_since_duration("7d").unwrap(), Duration::days(7));
        assert_eq!(parse_since_duration("2w").unwrap(), Duration::weeks(2));
        assert!(parse_since_duration("0d").is_err());
        assert!(parse_since_duration("soon").is_err());
    }

    #[test]
    fn since_overflow_is_a_clean_error_not_a_panic() {
        // Parseable but out-of-range amounts must not panic inside chrono.
        assert!(parse_since_duration("9999999999999w").is_err());
        assert!(parse_since_duration("9999999999999d").is_err());

        // A duration that fits in TimeDelta but overflows the now() subtraction
        // must also surface as a clean error rather than panicking.
        let command = ListCommand {
            json: false,
            jsonl: false,
            provider: None,
            local: false,
            cwd: None,
            since: Some("200000000d".to_string()),
            after: None,
            before: None,
            show_deleted_projects: false,
            limit: None,
        };
        assert!(list_after_cutoff(&command).is_err());
    }

    #[test]
    fn after_is_inclusive_and_before_is_exclusive() {
        let make = |id: &str, h: u32, m: u32| {
            let mut c = conversation(id, &format!("/tmp/{id}.jsonl"), ProviderKind::Claude);
            c.timestamp = Local
                .with_ymd_and_hms(2026, 6, 19, h, m, 0)
                .single()
                .unwrap();
            c
        };
        // Window is [09:00, 17:00): the conversation exactly on --after must stay,
        // the one exactly on --before must drop.
        let mut conversations = vec![
            make("before_after", 8, 59),
            make("on_after", 9, 0),
            make("inside", 12, 0),
            make("on_before", 17, 0),
        ];

        // Build RFC 3339 bounds in the local offset so the test is timezone-stable.
        let offset = Local
            .with_ymd_and_hms(2026, 6, 19, 0, 0, 0)
            .single()
            .unwrap()
            .format("%:z")
            .to_string();
        let command = ListCommand {
            json: false,
            jsonl: false,
            provider: None,
            local: false,
            cwd: None,
            since: None,
            after: Some(format!("2026-06-19T09:00:00{offset}")),
            before: Some(format!("2026-06-19T17:00:00{offset}")),
            show_deleted_projects: false,
            limit: None,
        };
        apply_list_filters(&mut conversations, &command).unwrap();

        let ids: Vec<_> = conversations.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["on_after", "inside"]);
    }

    #[test]
    fn cwd_filter_keeps_conversation_whose_recorded_dir_is_deleted() {
        // A recorded cwd that no longer exists on disk can't be canonicalized;
        // it must still match a --cwd root it is genuinely under (regression for
        // the /tmp -> /private/tmp symlink asymmetry on macOS).
        let root =
            std::env::temp_dir().join(format!("mnemonai-headless-cwd-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let gone = root.join("torn-down-worktree");

        let mut kept = conversation("kept", "/tmp/kept.jsonl", ProviderKind::Claude);
        kept.cwd = Some(gone.clone());
        kept.project_path = Some(gone.clone());

        let command = ListCommand {
            json: false,
            jsonl: false,
            provider: None,
            local: false,
            cwd: Some(root.clone()),
            since: None,
            after: None,
            before: None,
            show_deleted_projects: true,
            limit: None,
        };
        let mut conversations = vec![kept];
        apply_list_filters(&mut conversations, &command).unwrap();

        assert_eq!(
            conversations.len(),
            1,
            "deleted-but-under-root cwd was dropped"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn user_block_messages_get_block_index() {
        let entries = vec![LogEntry::User {
            message: UserMessage {
                content: UserContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "look at this".to_string(),
                    },
                    ContentBlock::Image {
                        source: serde_json::json!({"type": "base64"}),
                    },
                ]),
            },
            timestamp: "2026-06-19T10:00:00-07:00".to_string(),
            cwd: None,
        }];

        let messages = messages_from_entries(&entries);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].block_index, Some(0));
        assert_eq!(messages[1].role, "image");
        assert_eq!(messages[1].block_index, Some(1));
    }

    #[test]
    fn parses_timestamp_filters() {
        let date = parse_timestamp_filter("2026-06-20", "--after").unwrap();
        assert_eq!(date.date_naive().to_string(), "2026-06-20");
        assert_eq!(date.time().to_string(), "00:00:00");

        let timestamp = parse_timestamp_filter("2026-06-20T12:34:56-07:00", "--before").unwrap();
        // Compare instants, not local renderings: the parsed value is a
        // DateTime<Local>, so its RFC 3339 string depends on the host
        // timezone (this assertion must also hold on a UTC CI runner).
        let expected = chrono::DateTime::parse_from_rfc3339("2026-06-20T12:34:56-07:00").unwrap();
        assert_eq!(timestamp, expected);
        assert!(parse_timestamp_filter("06/20/2026", "--after").is_err());
    }

    #[test]
    fn list_filters_apply_time_window_and_cwd() {
        let root =
            std::env::temp_dir().join(format!("mnemonai-headless-filter-{}", std::process::id()));
        let subdir = root.join("subdir");
        let other = root.with_file_name(format!(
            "mnemonai-headless-filter-other-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::create_dir_all(&other).unwrap();

        let mut keep = conversation("keep", "/tmp/keep.jsonl", ProviderKind::Claude);
        keep.timestamp = Local
            .with_ymd_and_hms(2026, 6, 19, 12, 0, 0)
            .single()
            .unwrap();
        keep.cwd = Some(subdir.clone());
        keep.project_path = Some(root.clone());

        let mut too_old = conversation("old", "/tmp/old.jsonl", ProviderKind::Codex);
        too_old.timestamp = Local
            .with_ymd_and_hms(2026, 6, 10, 12, 0, 0)
            .single()
            .unwrap();
        too_old.cwd = Some(subdir.clone());
        too_old.project_path = Some(root.clone());

        let mut wrong_cwd = conversation("wrong", "/tmp/wrong.jsonl", ProviderKind::Cursor);
        wrong_cwd.timestamp = keep.timestamp;
        wrong_cwd.cwd = Some(other.clone());
        wrong_cwd.project_path = Some(other.clone());

        let command = ListCommand {
            json: false,
            jsonl: false,
            provider: None,
            local: false,
            cwd: Some(root.clone()),
            since: None,
            after: Some("2026-06-15".to_string()),
            before: Some("2026-06-20".to_string()),
            show_deleted_projects: false,
            limit: None,
        };
        let mut conversations = vec![keep, too_old, wrong_cwd];

        apply_list_filters(&mut conversations, &command).unwrap();

        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].id, "keep");

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(other);
    }

    #[test]
    fn headless_scope_is_global_without_explicit_local_flag() {
        let settings = HeadlessSettings {
            cli_local: false,
            cli_global: false,
            show_last: false,
            show_deleted_projects: false,
            debug: None,
        };

        assert!(
            !use_local_scope(&settings, false, false),
            "headless commands must default to global scope"
        );
        assert!(use_local_scope(&settings, true, false));

        let cli_settings = HeadlessSettings {
            cli_local: true,
            cli_global: false,
            show_last: false,
            show_deleted_projects: false,
            debug: None,
        };
        assert!(use_local_scope(&cli_settings, false, false));
        assert!(
            !use_local_scope(&cli_settings, true, true),
            "--cwd must force global loading before cwd filtering"
        );
    }

    #[test]
    fn global_flag_forces_global_scope_over_local() {
        // `--global` must override a subcommand `--local` so the flag's promise to
        // ignore directory scoping holds for `mnemonai --global list --local`.
        let settings = HeadlessSettings {
            cli_local: false,
            cli_global: true,
            show_last: false,
            show_deleted_projects: false,
            debug: None,
        };

        assert!(
            !use_local_scope(&settings, true, false),
            "--global must win over a subcommand --local"
        );
        assert!(!use_local_scope(&settings, false, false));
    }

    #[test]
    fn status_error_heuristic_is_conservative() {
        assert_eq!(status_is_error("failed"), Some(true));
        assert_eq!(status_is_error("error: exit 1"), Some(true));
        assert_eq!(status_is_error("completed"), Some(false));
        assert_eq!(status_is_error("success"), Some(false));
        assert_eq!(status_is_error("queued"), None);
        assert_eq!(status_is_error("incomplete"), None);
    }

    #[test]
    fn extracts_tool_result_exit_code_from_codex_output() {
        let text =
            "Chunk ID: abc\nWall time: 0.1 seconds\nProcess exited with code 101\nOutput:\nerror";

        assert_eq!(tool_result_exit_code(text), Some(101));
        assert_eq!(
            tool_result_exit_code("Process exited with code 0\nOutput:"),
            Some(0)
        );
        assert_eq!(
            tool_result_exit_code("Output:\n1\n2\n3\n4\n5\n6\n7\n8\nProcess exited with code 1"),
            None,
            "only provider metadata near the top of the result should be parsed"
        );
    }

    #[test]
    fn extracts_tool_result_exit_code_from_claude_output() {
        assert_eq!(
            tool_result_exit_code("Exit code 3\njq: syntax error"),
            Some(3)
        );
        assert_eq!(
            tool_result_exit_code("header\nExit code 1\nTraceback"),
            Some(1)
        );
        assert_eq!(
            tool_result_exit_code("Output:\n1\n2\n3\n4\n5\n6\n7\n8\nExit code 1"),
            None,
            "only provider metadata near the top of the result should be parsed"
        );
        assert_eq!(
            tool_result_exit_code("Exit code 2 indicates a syntax error in the input file."),
            None,
            "prose that merely starts with the marker phrase must not parse as an exit code"
        );
        assert_eq!(
            tool_result_exit_code("Exit code 1 (SIGHUP)"),
            None,
            "trailing annotations are not Claude's bare-marker format"
        );
        // Claude does not emit an "Exit code 0" success marker; when it does
        // appear as a bare line it still parses (downstream treats 0 as success).
        assert_eq!(tool_result_exit_code("Exit code 0"), Some(0));
    }

    #[test]
    fn merges_tool_result_error_markers_with_failure_precedence() {
        assert_eq!(merge_error_markers(&[None, None]), None);
        assert_eq!(merge_error_markers(&[Some(false), None]), Some(false));
        assert_eq!(merge_error_markers(&[Some(false), Some(true)]), Some(true));
    }

    #[test]
    fn infers_tool_result_error_from_exit_code() {
        let entries = vec![LogEntry::Assistant {
            message: AssistantMessage {
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "fail".to_string(),
                        content: Some(serde_json::Value::String(
                            "Chunk ID: abc\nWall time: 0.1 seconds\nProcess exited with code 101\nOutput:\ncompile failed"
                                .to_string(),
                        )),
                        is_error: None,
                        status: None,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "ok".to_string(),
                        content: Some(serde_json::Value::String(
                            "Chunk ID: def\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\ncompiled"
                                .to_string(),
                        )),
                        is_error: None,
                        status: None,
                    },
                ],
                model: None,
                usage: None,
                id: None,
            },
            timestamp: "2026-06-19T10:00:01-07:00".to_string(),
        }];

        let messages = messages_from_entries(&entries);

        assert_eq!(messages[0].tool_result_exit_code, Some(101));
        assert_eq!(messages[0].tool_result_error, Some(true));
        assert_eq!(messages[1].tool_result_exit_code, Some(0));
        assert_eq!(messages[1].tool_result_error, Some(false));
    }

    #[test]
    fn show_reads_and_normalizes_a_claude_jsonl_file() {
        use crate::providers::Provider;
        use crate::providers::claude::ClaudeProvider;

        let path = std::env::temp_dir().join(format!(
            "mnemonai-headless-show-{}.jsonl",
            std::process::id()
        ));
        let contents = concat!(
            r#"{"type":"user","message":{"role":"user","content":"hello there"},"timestamp":"2026-06-19T10:00:00-07:00"}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi back"}],"model":"claude-test"},"timestamp":"2026-06-19T10:00:01-07:00"}"#,
            "\n",
        );
        std::fs::write(&path, contents).unwrap();

        let conversation = conversation("rt-id", path.to_str().unwrap(), ProviderKind::Claude);
        let provider = ClaudeProvider::new(vec![]);
        let entries = provider.read_entries(&conversation).unwrap();
        let messages = messages_from_entries(&entries);

        std::fs::remove_file(&path).ok();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].text.as_deref(), Some("hello there"));
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].text.as_deref(), Some("hi back"));
        assert_eq!(messages[1].model.as_deref(), Some("claude-test"));
    }

    // ---- search ----

    fn conv_text(
        id: &str,
        provider: ProviderKind,
        text: &str,
        timestamp: DateTime<Local>,
    ) -> Conversation {
        Conversation {
            path: PathBuf::from(format!("/tmp/{id}.jsonl")),
            provider,
            id: id.to_string(),
            timestamp,
            preview: text.chars().take(40).collect(),
            full_text: text.to_string(),
            project_name: Some("proj".to_string()),
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: Vec::new(),
            summary: Some(format!("summary {id}")),
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    fn search_command(words: &[&str]) -> SearchCommand {
        SearchCommand {
            words: words.iter().map(|w| w.to_string()).collect(),
            json: false,
            jsonl: false,
            provider: None,
            local: false,
            cwd: None,
            since: None,
            after: None,
            before: None,
            limit: 10,
            snippets: 2,
            exclude_session: Vec::new(),
        }
    }

    #[test]
    fn headless_search_order_matches_tui_search() {
        // The headless ranking must reproduce the interactive list order exactly.
        let now = Local.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let convs = vec![
            conv_text(
                "dense",
                ProviderKind::Claude,
                "deploy deploy deploy the deploy fix for deploy",
                now,
            ),
            conv_text(
                "sparse",
                ProviderKind::Codex,
                &format!("deploy once {}", "filler ".repeat(120)),
                now - Duration::days(1),
            ),
            conv_text("none", ProviderKind::Cursor, "nothing to see here", now),
        ];

        let command = search_command(&["deploy"]);
        let headless_ids: Vec<String> = rank_and_build(convs.clone(), &command, now)
            .into_iter()
            .map(|result| result.id)
            .collect();

        let mut tui_convs = convs.clone();
        let searchable = search::precompute_search_text(&mut tui_convs);
        let tui_ids: Vec<String> = search::search(&tui_convs, &searchable, "deploy", now, None)
            .into_iter()
            .map(|index| tui_convs[index].id.clone())
            .collect();

        assert_eq!(headless_ids, tui_ids);
        assert_eq!(headless_ids, vec!["dense", "sparse"]);

        // Multi-word: the CLI joins WORDS with single spaces, so ranking must
        // match a TUI user typing the same words (no trailing space). "fi" is a
        // substring of both "fix" (dense) and "filler" (sparse); AND drops "none".
        let command = search_command(&["deploy", "fi"]);
        let headless_multi: Vec<String> = rank_and_build(convs.clone(), &command, now)
            .into_iter()
            .map(|result| result.id)
            .collect();
        let tui_multi: Vec<String> =
            search::search(&tui_convs, &searchable, "deploy fi", now, None)
                .into_iter()
                .map(|index| tui_convs[index].id.clone())
                .collect();

        assert_eq!(headless_multi, tui_multi);
        assert_eq!(headless_multi.len(), 2);
        assert!(!headless_multi.contains(&"none".to_string()));
    }

    #[test]
    fn search_joins_words_as_substring_terms_like_typeahead() {
        let now = Local::now();
        let convs = vec![
            conv_text(
                "both",
                ProviderKind::Claude,
                "reviewing the workspace flowers today",
                now,
            ),
            conv_text(
                "one",
                ProviderKind::Codex,
                "only workspace here, no petals",
                now,
            ),
        ];
        // The join has no trailing space, so no term is "completed": interior
        // "work" and final "flow" both match as substrings (workspace, flowers),
        // and AND logic requires both to appear.
        let mut command = search_command(&["work", "flow"]);
        command.snippets = 0;
        let ids: Vec<String> = rank_and_build(convs, &command, now)
            .into_iter()
            .map(|result| result.id)
            .collect();
        assert_eq!(ids, vec!["both"]);
    }

    #[test]
    fn search_snippets_are_lowercased_bounded_and_clamped() {
        let now = Local::now();
        let filler = "x".repeat(300);
        let text = format!(
            "JOB_WORKFLOW_REF start {filler} middle JOB_WORKFLOW_REF here {filler} tail JOB_WORKFLOW_REF end"
        );
        let convs = vec![conv_text("s", ProviderKind::Claude, &text, now)];

        let mut command = search_command(&["job_workflow_ref"]);
        command.snippets = 2;
        let results = rank_and_build(convs.clone(), &command, now);
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.match_count, 3, "case-insensitive occurrence count");
        assert_eq!(result.snippets.len(), 2, "clamped to the requested count");
        for snippet in &result.snippets {
            assert_eq!(
                snippet,
                &snippet.to_lowercase(),
                "snippets come from the lowercased corpus"
            );
            assert!(
                snippet.contains("job_workflow_ref"),
                "each snippet is centered on a match: {snippet}"
            );
            assert!(
                snippet.chars().count() <= SNIPPET_WIDTH + 4,
                "snippet stays near the window width: {} chars",
                snippet.chars().count()
            );
        }

        command.snippets = 0;
        assert!(
            rank_and_build(convs.clone(), &command, now)[0]
                .snippets
                .is_empty()
        );

        command.snippets = 9;
        let clamped = rank_and_build(convs, &command, now);
        assert!(clamped[0].snippets.len() <= SNIPPET_MAX);
        assert_eq!(
            clamped[0].snippets.len(),
            3,
            "only three distinct windows exist"
        );
    }

    #[test]
    fn search_snippets_slice_multibyte_text_without_panicking() {
        // Snippet windows are centered on byte offsets inside CJK/emoji text;
        // slicing must land on char boundaries instead of panicking mid-codepoint.
        let now = Local::now();
        let text = format!(
            "{} deploy {} deploy {}",
            "汉字🎉".repeat(80),
            "émoji🚀".repeat(80),
            "日本語".repeat(80)
        );
        let convs = vec![conv_text("mb", ProviderKind::Claude, &text, now)];

        let mut command = search_command(&["deploy"]);
        command.snippets = 5;
        let results = rank_and_build(convs, &command, now);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].match_count, 2);
        assert_eq!(results[0].snippets.len(), 2, "matches are >240 bytes apart");
        for snippet in &results[0].snippets {
            assert!(snippet.contains("deploy"), "snippet centers a match");
        }
    }

    #[test]
    fn search_composes_provider_since_query_and_exclude() {
        let now = Local::now();
        let providers: Vec<Box<dyn Provider>> = vec![
            StubProvider::ok(
                ProviderKind::Claude,
                vec![conv_text(
                    "claude-hit",
                    ProviderKind::Claude,
                    "workflow secrets rotation",
                    now,
                )],
            ),
            StubProvider::ok(
                ProviderKind::Codex,
                vec![
                    conv_text(
                        "codex-recent",
                        ProviderKind::Codex,
                        "reusable workflow secrets",
                        now,
                    ),
                    conv_text(
                        "codex-old",
                        ProviderKind::Codex,
                        "reusable workflow secrets",
                        now - Duration::days(120),
                    ),
                    conv_text(
                        "codex-excluded",
                        ProviderKind::Codex,
                        "reusable workflow secrets",
                        now,
                    ),
                ],
            ),
        ];
        let settings = HeadlessSettings {
            cli_local: false,
            cli_global: false,
            show_last: false,
            show_deleted_projects: false,
            debug: None,
        };

        // Provider filter keeps only Codex; full text is requested for ranking.
        let mut conversations = load_conversations(
            &providers,
            &settings,
            Some(ProviderFilter::Codex),
            false,
            false,
            false,
            true,
        )
        .unwrap();
        let loaded: Vec<&str> = conversations.iter().map(|c| c.id.as_str()).collect();
        assert!(
            !loaded.contains(&"claude-hit"),
            "provider filter drops Claude"
        );
        assert!(loaded.contains(&"codex-recent"));

        let mut command = search_command(&["workflow", "secret"]);
        command.provider = Some(ProviderFilter::Codex);
        command.since = Some("30d".to_string());
        command.exclude_session = vec!["codex-excluded".to_string()];
        command.snippets = 0;

        // --since drops the 120-day-old conversation before ranking.
        apply_search_filters(&mut conversations, &command).unwrap();
        let result_ids: Vec<String> = rank_and_build(conversations, &command, now)
            .into_iter()
            .map(|result| result.id)
            .collect();
        assert_eq!(result_ids, vec!["codex-recent"]);
    }

    #[test]
    fn search_result_json_shape_is_pinned() {
        let now = Local::now();
        let convs = vec![conv_text(
            "sess-1",
            ProviderKind::CursorAgent,
            "job_workflow_ref discussion here",
            now,
        )];
        let mut command = search_command(&["job_workflow_ref"]);
        command.snippets = 1;
        let results = rank_and_build(convs, &command, now);
        let json = serde_json::to_value(&results).unwrap();
        let obj = &json[0];

        assert_eq!(obj["provider"], "cursor-agent");
        assert_eq!(obj["id"], "sess-1");
        assert_eq!(obj["path"], "/tmp/sess-1.jsonl");
        assert!(obj["timestamp"].is_string());
        assert_eq!(obj["project_name"], "proj");
        assert!(obj["cwd"].is_null());
        assert_eq!(obj["match_count"], 1);
        assert!(obj["score"].is_number());
        assert_eq!(obj["snippets"].as_array().unwrap().len(), 1);

        // Slim by design: no preview, no parse_errors.
        assert!(obj.get("preview").is_none());
        assert!(obj.get("parse_errors").is_none());
        for key in [
            "provider",
            "id",
            "path",
            "timestamp",
            "project_name",
            "cwd",
            "summary",
            "score",
            "match_count",
            "snippets",
        ] {
            assert!(obj.get(key).is_some(), "missing key {key}");
        }
    }

    #[test]
    fn search_empty_results_serialize_as_empty_array() {
        let results: Vec<SearchResult> = Vec::new();
        let bytes = json_bytes(&results).unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), "[]\n");
    }

    // ---- show --grep ----

    fn text_msg(index: usize, text: &str) -> MessageDto {
        let mut message = MessageDto::new("user", MessageContext::new(index));
        message.index = index;
        message.text = Some(text.to_string());
        message
    }

    fn tool_result_msg(index: usize, result: Value) -> MessageDto {
        let mut message = MessageDto::new("tool_result", MessageContext::new(index));
        message.index = index;
        message.tool_result = Some(result);
        message
    }

    #[test]
    fn grep_or_matches_tool_fields_and_merges_overlapping_context() {
        let messages = vec![
            text_msg(0, "intro line"),
            text_msg(1, "the ALPHA appears here"),
            tool_result_msg(2, serde_json::json!({"stdout": "contains beta token"})),
            text_msg(3, "trailing note"),
            text_msg(4, "far away tail"),
        ];
        // OR of two patterns: "alpha" hits message 1 (text), "beta" hits message 2
        // (inside the tool_result JSON). Context windows [0,2] and [1,3] overlap
        // and merge; message 4 is excluded.
        let filtered = apply_grep(messages, &["alpha".to_string(), "beta".to_string()], 1);

        let indices: Vec<usize> = filtered.iter().map(|m| m.index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);

        let matched: Vec<usize> = filtered
            .iter()
            .filter(|m| m.matched == Some(true))
            .map(|m| m.index)
            .collect();
        assert_eq!(
            matched,
            vec![1, 2],
            "only real matches carry the matched flag"
        );
        assert_eq!(
            filtered[0].matched, None,
            "context-only neighbor has no flag"
        );
        assert_eq!(filtered[3].matched, None);
    }

    #[test]
    fn grep_merges_adjacent_context_windows() {
        let messages = vec![
            text_msg(0, "a"),
            text_msg(1, "alpha"),
            text_msg(2, "b"),
            text_msg(3, "c"),
            text_msg(4, "alpha"),
            text_msg(5, "d"),
        ];
        // Windows [0,2] and [3,5] are adjacent and merge into one contiguous run.
        let filtered = apply_grep(messages, &["alpha".to_string()], 1);
        let indices: Vec<usize> = filtered.iter().map(|m| m.index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
        let matched: Vec<usize> = filtered
            .iter()
            .filter(|m| m.matched == Some(true))
            .map(|m| m.index)
            .collect();
        assert_eq!(matched, vec![1, 4]);
    }

    #[test]
    fn grep_with_no_hits_returns_empty() {
        let messages = vec![text_msg(0, "nothing"), text_msg(1, "here")];
        assert!(apply_grep(messages, &["zzz".to_string()], 2).is_empty());
    }

    #[test]
    fn grep_detail_json_includes_total_messages_and_matched() {
        let messages = vec![
            text_msg(0, "intro"),
            text_msg(1, "alpha match"),
            text_msg(2, "beta"),
        ];
        let total = messages.len();
        let filtered = apply_grep(messages, &["alpha".to_string()], 0);
        let detail = ConversationDetail {
            conversation: ConversationSummary::from_conversation(&conversation(
                "c",
                "/tmp/c.jsonl",
                ProviderKind::Claude,
            )),
            total_messages: Some(total),
            messages: filtered,
        };
        let json = serde_json::to_value(&detail).unwrap();
        assert_eq!(json["total_messages"], 3);
        assert_eq!(json["messages"].as_array().unwrap().len(), 1);
        assert_eq!(json["messages"][0]["matched"], true);
    }

    #[test]
    fn plain_show_output_is_byte_identical_without_new_fields() {
        // Without --grep, ConversationDetail carries total_messages = None and no
        // message is flagged, so the serialization must match the historical
        // {conversation, messages} shape byte for byte.
        #[derive(Serialize)]
        struct OldDetail {
            conversation: ConversationSummary,
            messages: Vec<MessageDto>,
        }

        let entries = vec![
            LogEntry::User {
                message: UserMessage {
                    content: UserContent::String("run tests".to_string()),
                },
                timestamp: "2026-06-19T10:00:00-07:00".to_string(),
                cwd: None,
            },
            LogEntry::Assistant {
                message: AssistantMessage {
                    content: vec![ContentBlock::Text {
                        text: "on it".to_string(),
                    }],
                    model: Some("claude-test".to_string()),
                    usage: None,
                    id: None,
                },
                timestamp: "2026-06-19T10:00:01-07:00".to_string(),
            },
        ];
        let conv = conversation("byte-id", "/tmp/byte-id.jsonl", ProviderKind::Claude);

        let new_detail = ConversationDetail {
            conversation: ConversationSummary::from_conversation(&conv),
            total_messages: None,
            messages: messages_from_entries(&entries),
        };
        let old_detail = OldDetail {
            conversation: ConversationSummary::from_conversation(&conv),
            messages: messages_from_entries(&entries),
        };

        let new_bytes = json_bytes(&new_detail).unwrap();
        assert_eq!(new_bytes, json_bytes(&old_detail).unwrap());

        let rendered = String::from_utf8(new_bytes).unwrap();
        assert!(!rendered.contains("total_messages"));
        assert!(!rendered.contains("\"matched\""));
    }
}
