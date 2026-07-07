use crate::claude::{
    AgentContent, ContentBlock, LogEntry, UserContent, extract_tool_result_text,
    parse_agent_progress,
};
use crate::cli::{
    Command, DebugLevel, ExcerptCommand, ListCommand, ProviderFilter, SearchCommand, ShowCommand,
};
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, ParseError, path_to_string, project_path_is_live,
};
use crate::providers::Provider;
use crate::tui::search::{normalize_for_search, precompute_search_text, search_with_scores};
use chrono::{DateTime, Duration, Local, NaiveDate, TimeZone};
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::Path;

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
    messages: Vec<MessageDto>,
}

#[derive(Serialize)]
struct SearchConversationSummary {
    score: f64,
    #[serde(flatten)]
    conversation: ConversationSummary,
}

#[derive(Serialize)]
struct ConversationExcerpt {
    conversation: ConversationSummary,
    windows: Vec<ExcerptWindow>,
}

#[derive(Serialize)]
struct ExcerptWindow {
    start_index: usize,
    end_index: usize,
    messages: Vec<MessageDto>,
}

#[derive(Clone, Serialize)]
struct MessageDto {
    index: usize,
    entry_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_index: Option<usize>,
    role: String,
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
        Command::Excerpt(command) => run_excerpt(command, providers, settings),
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
    )?;
    apply_list_filters(&mut conversations, command)?;
    if let Some(limit) = command.limit {
        conversations.truncate(limit);
    }
    reindex_conversations(&mut conversations);

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
    let detail = ConversationDetail {
        conversation: ConversationSummary::from_conversation(conversation),
        messages: messages_from_entries(&entries),
    };

    write_json(&detail)
}

fn run_search(
    command: &SearchCommand,
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
) -> Result<()> {
    if command.query.trim().is_empty() {
        return Err(AppError::CommandError(
            "Search query must not be empty".to_string(),
        ));
    }

    let mut conversations = load_conversations(
        providers,
        settings,
        command.provider,
        command.local,
        command.show_deleted_projects,
        command.cwd.is_some(),
    )?;
    apply_search_filters(&mut conversations, command)?;
    let results = search_conversations(conversations, &command.query, command.limit);

    if command.jsonl {
        write_jsonl(&results)
    } else {
        write_json(&results)
    }
}

fn run_excerpt(
    command: &ExcerptCommand,
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
    let windows = excerpt_windows_from_messages(&messages, command)?;
    let excerpt = ConversationExcerpt {
        conversation: ConversationSummary::from_conversation(conversation),
        windows,
    };

    if command.markdown {
        let rendered = render_excerpt_markdown(&excerpt);
        write_stdout(rendered.as_bytes())
    } else {
        write_json(&excerpt)
    }
}

fn load_conversations(
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
    provider_filter: Option<ProviderFilter>,
    command_local: bool,
    command_show_deleted: bool,
    force_global: bool,
) -> Result<Vec<Conversation>> {
    let local = use_local_scope(settings, command_local, force_global);
    let show_deleted_projects = command_show_deleted || settings.show_deleted_projects;
    let mut conversations = if local {
        crate::loader::load_local(
            providers,
            settings.show_last,
            settings.debug,
            provider_filter,
        )?
    } else {
        load_global_conversations(
            providers,
            settings.show_last,
            settings.debug,
            provider_filter,
        )?
    };

    if !show_deleted_projects {
        conversations.retain(|conversation| {
            conversation
                .project_path
                .as_ref()
                .is_none_or(|path| project_path_is_live(path))
        });
    }

    conversations.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    reindex_conversations(&mut conversations);

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

fn apply_search_filters(
    conversations: &mut Vec<Conversation>,
    command: &SearchCommand,
) -> Result<()> {
    let list_command = ListCommand {
        json: false,
        jsonl: false,
        provider: command.provider,
        local: command.local,
        cwd: command.cwd.clone(),
        since: command.since.clone(),
        after: command.after.clone(),
        before: command.before.clone(),
        show_deleted_projects: command.show_deleted_projects,
        limit: command.limit,
    };
    apply_list_filters(conversations, &list_command)
}

fn search_conversations(
    mut conversations: Vec<Conversation>,
    query: &str,
    limit: Option<usize>,
) -> Vec<SearchConversationSummary> {
    let searchable = precompute_search_text(&mut conversations);
    let mut ranked = search_with_scores(&conversations, &searchable, query, Local::now(), None);
    if let Some(limit) = limit {
        ranked.truncate(limit);
    }

    ranked
        .into_iter()
        .map(|result| SearchConversationSummary {
            score: result.score,
            conversation: ConversationSummary::from_conversation(&conversations[result.index]),
        })
        .collect()
}

fn list_after_cutoff(command: &ListCommand) -> Result<Option<DateTime<Local>>> {
    if let Some(since) = command.since.as_deref() {
        let duration = parse_since_duration(since)?;
        let cutoff = Local::now()
            .checked_sub_signed(duration)
            .ok_or_else(|| invalid_duration(since))?;
        Ok(Some(cutoff))
    } else {
        command
            .after
            .as_deref()
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

fn reindex_conversations(conversations: &mut [Conversation]) {
    for (idx, conversation) in conversations.iter_mut().enumerate() {
        conversation.index = idx;
    }
}

fn load_global_conversations(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug: Option<DebugLevel>,
    provider_filter: Option<ProviderFilter>,
) -> Result<Vec<Conversation>> {
    let mut conversations = Vec::new();

    for provider in providers {
        if !crate::loader::provider_filter_matches(provider_filter, &provider.kind()) {
            continue;
        }

        let rx = provider.load_conversations_streaming(show_last, debug);
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
    let target_canonical = target_path.canonicalize().ok();

    let matches: Vec<&Conversation> = conversations
        .iter()
        .filter(|conversation| {
            path_matches(&conversation.path, target_path, target_canonical.as_deref())
                || conversation.id == target
                || conversation
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem == target)
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
    path == target
        || target_canonical.is_some_and(|canonical| {
            path.canonicalize()
                .map(|path| path == canonical)
                .unwrap_or(false)
        })
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
                message,
                timestamp,
                cwd: _,
                ..
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

fn excerpt_windows_from_messages(
    messages: &[MessageDto],
    command: &ExcerptCommand,
) -> Result<Vec<ExcerptWindow>> {
    if let Some(query) = command.query.as_deref() {
        if query.trim().is_empty() {
            return Err(AppError::CommandError(
                "Excerpt query must not be empty".to_string(),
            ));
        }
        let match_roles = parse_match_roles(command.match_roles.as_deref())?;
        let mut ranges =
            query_excerpt_ranges(messages, query, command.context, match_roles.as_deref());
        if let Some(max_windows) = command.max_windows {
            ranges.truncate(max_windows);
        }
        return Ok(ranges
            .into_iter()
            .map(|(start, end)| excerpt_window(messages, start, end))
            .collect());
    }

    let Some(start) = command.from_index else {
        return Err(AppError::CommandError(
            "Pass --query <text> or --from <index> to select excerpt messages".to_string(),
        ));
    };
    let end = command.to_index.unwrap_or(start);
    validate_message_range(messages, start, end)?;
    Ok(vec![excerpt_window(messages, start, end)])
}

fn validate_message_range(messages: &[MessageDto], start: usize, end: usize) -> Result<()> {
    if start > end {
        return Err(AppError::CommandError(format!(
            "Invalid excerpt range: --from {start} is greater than --to {end}"
        )));
    }
    if messages.is_empty() {
        return Err(AppError::CommandError(
            "Cannot excerpt an empty conversation".to_string(),
        ));
    }
    if end >= messages.len() {
        return Err(AppError::CommandError(format!(
            "Excerpt range {start}-{end} is outside message index range 0-{}",
            messages.len() - 1
        )));
    }
    Ok(())
}

fn excerpt_window(messages: &[MessageDto], start: usize, end: usize) -> ExcerptWindow {
    ExcerptWindow {
        start_index: start,
        end_index: end,
        messages: messages[start..=end].to_vec(),
    }
}

fn query_excerpt_ranges(
    messages: &[MessageDto],
    query: &str,
    context: usize,
    match_roles: Option<&[String]>,
) -> Vec<(usize, usize)> {
    if messages.is_empty() {
        return Vec::new();
    }

    let terms: Vec<_> = normalize_for_search(query)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }

    let message_texts: Vec<_> = messages
        .iter()
        .map(|message| normalize_for_search(&message_search_text(message)))
        .collect();
    let hits_by_term: Vec<Vec<usize>> = terms
        .iter()
        .map(|term| {
            message_texts
                .iter()
                .enumerate()
                .filter_map(|(idx, text)| {
                    (message_role_matches(match_roles, &messages[idx].role) && text.contains(term))
                        .then_some(idx)
                })
                .collect()
        })
        .collect();

    if hits_by_term.iter().any(Vec::is_empty) {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    for center in 0..messages.len() {
        let search_start = center.saturating_sub(context);
        let search_end = center.saturating_add(context).min(messages.len() - 1);
        let mut min_hit = usize::MAX;
        let mut max_hit = 0;
        let mut matched_all_terms = true;

        for hits in &hits_by_term {
            let Some(hit) = hits
                .iter()
                .copied()
                .find(|idx| *idx >= search_start && *idx <= search_end)
            else {
                matched_all_terms = false;
                break;
            };
            min_hit = min_hit.min(hit);
            max_hit = max_hit.max(hit);
        }

        if matched_all_terms {
            ranges.push((
                min_hit.saturating_sub(context),
                max_hit.saturating_add(context).min(messages.len() - 1),
            ));
        }
    }

    merge_ranges(ranges)
}

fn parse_match_roles(value: Option<&str>) -> Result<Option<Vec<String>>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let roles: Vec<_> = value
        .split(',')
        .map(normalize_match_role)
        .filter(|role| !role.is_empty())
        .collect();

    if roles.is_empty() {
        return Err(AppError::CommandError(
            "Invalid --match-roles value; expected comma-separated roles like user,assistant"
                .to_string(),
        ));
    }

    Ok(Some(roles))
}

fn normalize_match_role(role: &str) -> String {
    role.trim().to_ascii_lowercase().replace('-', "_")
}

fn message_role_matches(match_roles: Option<&[String]>, role: &str) -> bool {
    let Some(match_roles) = match_roles else {
        return true;
    };
    let role = normalize_match_role(role);
    match_roles.iter().any(|candidate| {
        candidate == "*"
            || candidate == &role
            || (candidate == "agent_*" && role.starts_with("agent_"))
    })
}

fn message_search_text(message: &MessageDto) -> String {
    let mut text = String::new();
    push_search_part(&mut text, &message.role);
    if let Some(value) = message.text.as_deref() {
        push_search_part(&mut text, value);
    }
    if let Some(value) = message.thinking.as_deref() {
        push_search_part(&mut text, value);
    }
    if let Some(value) = message.tool_name.as_deref() {
        push_search_part(&mut text, value);
    }
    if let Some(value) = message.tool_input.as_ref() {
        push_search_part(&mut text, &value.to_string());
    }
    if let Some(value) = message.tool_result_status.as_deref() {
        push_search_part(&mut text, value);
    }
    if let Some(value) = message.tool_result_exit_code {
        push_search_part(&mut text, &value.to_string());
    }
    if let Some(value) = message.subtype.as_deref() {
        push_search_part(&mut text, value);
    }
    if let Some(value) = message.level.as_deref() {
        push_search_part(&mut text, value);
    }
    text
}

fn push_search_part(text: &mut String, value: &str) {
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(value);
}

fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_unstable();
    let mut merged = Vec::new();
    let mut current = ranges[0];
    for range in ranges.into_iter().skip(1) {
        if range.0 <= current.1.saturating_add(1) {
            current.1 = current.1.max(range.1);
        } else {
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);
    merged
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

fn render_excerpt_markdown(excerpt: &ConversationExcerpt) -> String {
    let conversation = &excerpt.conversation;
    let mut output = String::new();
    output.push_str("# mnemonai excerpt\n\n");
    output.push_str(&format!("- Provider: {}\n", conversation.provider));
    output.push_str(&format!("- Session: {}\n", conversation.id));
    output.push_str(&format!("- Path: {}\n", conversation.path));
    output.push_str(&format!("- Timestamp: {}\n", conversation.timestamp));
    if let Some(project_name) = conversation.project_name.as_deref() {
        output.push_str(&format!("- Project: {project_name}\n"));
    }
    if let Some(project_path) = conversation.project_path.as_deref() {
        output.push_str(&format!("- Project path: {project_path}\n"));
    }
    if let Some(cwd) = conversation.cwd.as_deref() {
        output.push_str(&format!("- Cwd: {cwd}\n"));
    }
    if let Some(summary) = conversation.summary.as_deref() {
        output.push_str(&format!("- Summary: {summary}\n"));
    }
    output.push('\n');

    if excerpt.windows.is_empty() {
        output.push_str("No matching excerpts.\n");
        return output;
    }

    for window in &excerpt.windows {
        output.push_str(&format!(
            "## Messages {}-{}\n\n",
            window.start_index, window.end_index
        ));
        for message in &window.messages {
            render_message_markdown(&mut output, message);
        }
    }

    output
}

fn render_message_markdown(output: &mut String, message: &MessageDto) {
    output.push_str(&format!("### [{}] {}", message.index, message.role));
    if let Some(timestamp) = message.timestamp.as_deref() {
        output.push_str(&format!(" ({timestamp})"));
    }
    output.push_str("\n\n");

    if let Some(text) = message.text.as_deref() {
        push_fenced_block(output, "text", text);
    }
    if let Some(thinking) = message.thinking.as_deref() {
        push_fenced_block(output, "thinking", thinking);
    }
    if let Some(tool_name) = message.tool_name.as_deref() {
        output.push_str(&format!("- Tool: {tool_name}\n"));
    }
    if let Some(tool_call_id) = message.tool_call_id.as_deref() {
        output.push_str(&format!("- Tool call ID: {tool_call_id}\n"));
    }
    if let Some(tool_input) = message.tool_input.as_ref() {
        push_json_block(output, "tool_input", tool_input);
    }
    if let Some(tool_result_status) = message.tool_result_status.as_deref() {
        output.push_str(&format!("- Tool result status: {tool_result_status}\n"));
    }
    if let Some(tool_result_exit_code) = message.tool_result_exit_code {
        output.push_str(&format!(
            "- Tool result exit code: {tool_result_exit_code}\n"
        ));
    }
    if let Some(tool_result_error) = message.tool_result_error {
        output.push_str(&format!("- Tool result error: {tool_result_error}\n"));
    }
    if let Some(tool_result) = message.tool_result.as_ref() {
        push_json_block(output, "tool_result", tool_result);
    }
    if let Some(agent_id) = message.agent_id.as_deref() {
        output.push_str(&format!("- Agent ID: {agent_id}\n"));
    }
    if let Some(subtype) = message.subtype.as_deref() {
        output.push_str(&format!("- Subtype: {subtype}\n"));
    }
    if let Some(level) = message.level.as_deref() {
        output.push_str(&format!("- Level: {level}\n"));
    }
    if let Some(duration_ms) = message.duration_ms {
        output.push_str(&format!("- Duration ms: {duration_ms}\n"));
    }
    output.push('\n');
}

fn push_json_block(output: &mut String, label: &str, value: &Value) {
    let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    output.push_str(&format!("{label}:\n\n"));
    push_fenced_block(output, "json", &rendered);
}

fn push_fenced_block(output: &mut String, language: &str, text: &str) {
    let fence = if text.contains("```") { "````" } else { "```" };
    output.push_str(&format!("{fence}{language}\n{text}\n{fence}\n\n"));
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
            index: 0,
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
            search_text_lower: None,
            search_topic_end: None,
        }
    }

    fn text_message(index: usize, role: &str, text: &str) -> MessageDto {
        let mut message = MessageDto::new(role, MessageContext::new(index));
        message.index = index;
        message.text = Some(text.to_string());
        message
    }

    fn excerpt_command(
        query: Option<&str>,
        from_index: Option<usize>,
        to_index: Option<usize>,
        context: usize,
    ) -> ExcerptCommand {
        ExcerptCommand {
            target: "session".to_string(),
            query: query.map(ToString::to_string),
            from_index,
            to_index,
            context,
            match_roles: None,
            max_windows: None,
            json: false,
            markdown: false,
            provider: None,
            local: false,
            show_deleted_projects: false,
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
    fn search_conversations_returns_ranked_summaries() {
        let mut sparse = conversation("sparse", "/tmp/sparse.jsonl", ProviderKind::Claude);
        sparse.full_text = format!("we might deploy once {}", "filler ".repeat(200));

        let mut dense = conversation("dense", "/tmp/dense.jsonl", ProviderKind::Codex);
        dense.full_text = "deploy deploy deploy release deploy".to_string();

        let results = search_conversations(vec![sparse, dense], "deploy", Some(1));

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conversation.id, "dense");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn explicit_excerpt_range_keeps_message_indices() {
        let messages = vec![
            text_message(0, "user", "zero"),
            text_message(1, "assistant", "one"),
            text_message(2, "user", "two"),
        ];
        let command = excerpt_command(None, Some(1), Some(2), 3);

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_index, 1);
        assert_eq!(windows[0].end_index, 2);
        assert_eq!(windows[0].messages[0].index, 1);
        assert_eq!(windows[0].messages[1].index, 2);
    }

    #[test]
    fn query_excerpt_matches_terms_across_nearby_messages() {
        let messages = vec![
            text_message(0, "user", "notes about emp tower"),
            text_message(1, "assistant", "intermediate detail"),
            text_message(2, "user", "SP work in cnc3"),
            text_message(3, "assistant", "follow-up"),
        ];
        let command = excerpt_command(Some("emp tower SP cnc3"), None, None, 1);

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_index, 0);
        assert_eq!(windows[0].end_index, 3);
    }

    #[test]
    fn query_excerpt_match_roles_ignore_tool_result_noise() {
        let messages = vec![
            text_message(
                0,
                "tool_result",
                "emp tower SP cnc3 appears in noisy output",
            ),
            text_message(1, "assistant", "unrelated narrative"),
            text_message(2, "user", "still unrelated"),
        ];
        let mut command = excerpt_command(Some("emp tower SP cnc3"), None, None, 1);
        command.match_roles = Some("user,assistant,summary".to_string());

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert!(
            windows.is_empty(),
            "tool_result text should not trigger excerpts when role matching excludes it"
        );
    }

    #[test]
    fn query_excerpt_match_roles_allow_hyphenated_role_aliases() {
        let messages = vec![
            text_message(0, "tool_result", "emp tower"),
            text_message(1, "assistant", "SP in cnc3"),
        ];
        let mut command = excerpt_command(Some("emp tower"), None, None, 1);
        command.match_roles = Some("tool-result".to_string());

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_index, 0);
        assert_eq!(windows[0].end_index, 1);
    }

    #[test]
    fn query_excerpt_max_windows_caps_merged_ranges() {
        let messages = vec![
            text_message(0, "user", "first match"),
            text_message(1, "assistant", "gap"),
            text_message(2, "assistant", "gap"),
            text_message(3, "user", "second match"),
            text_message(4, "assistant", "gap"),
            text_message(5, "assistant", "gap"),
            text_message(6, "user", "third match"),
        ];
        let mut command = excerpt_command(Some("match"), None, None, 0);
        command.max_windows = Some(2);

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_index, 0);
        assert_eq!(windows[1].start_index, 3);
    }

    #[test]
    fn query_excerpt_returns_empty_windows_when_no_match() {
        let messages = vec![text_message(0, "user", "unrelated")];
        let command = excerpt_command(Some("emp tower SP"), None, None, 2);

        let windows = excerpt_windows_from_messages(&messages, &command).unwrap();

        assert!(windows.is_empty());
    }

    #[test]
    fn markdown_excerpt_renders_source_and_message_range() {
        let conversation = ConversationSummary::from_conversation(&conversation(
            "abc",
            "/tmp/abc.jsonl",
            ProviderKind::Codex,
        ));
        let excerpt = ConversationExcerpt {
            conversation,
            windows: vec![ExcerptWindow {
                start_index: 0,
                end_index: 0,
                messages: vec![text_message(0, "user", "hello")],
            }],
        };

        let rendered = render_excerpt_markdown(&excerpt);

        assert!(rendered.contains("- Provider: codex"));
        assert!(rendered.contains("- Session: abc"));
        assert!(rendered.contains("## Messages 0-0"));
        assert!(rendered.contains("```text\nhello\n```"));
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
        assert_eq!(timestamp.to_rfc3339(), "2026-06-20T12:34:56-07:00");
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
}
