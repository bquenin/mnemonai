use crate::claude::{AgentContent, ContentBlock, LogEntry, UserContent, parse_agent_progress};
use crate::cli::{Command, DebugLevel, ListCommand, ProviderFilter, ShowCommand};
use crate::error::{AppError, Result};
use crate::history::{Conversation, LoaderMessage, ParseError, ProviderKind, project_path_is_live};
use crate::providers::Provider;
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct HeadlessSettings {
    pub cli_local: bool,
    pub config_local: bool,
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
struct MessageDto {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<Value>,
}

impl MessageDto {
    fn new(role: impl Into<String>, timestamp: Option<&str>) -> Self {
        Self {
            role: role.into(),
            timestamp: timestamp.map(ToString::to_string),
            text: None,
            tool_name: None,
            tool_input: None,
            tool_result: None,
            thinking: None,
            model: None,
            agent_id: None,
            subtype: None,
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
    )?;
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
    )?;
    let conversation = resolve_conversation(&conversations, &command.target)?;
    let provider = providers
        .iter()
        .find(|provider| provider.kind() == conversation.provider)
        .ok_or_else(|| {
            AppError::CommandError(format!(
                "No provider found for {} conversation",
                provider_key(&conversation.provider)
            ))
        })?;
    let entries = provider.read_entries(conversation)?;
    let detail = ConversationDetail {
        conversation: ConversationSummary::from_conversation(conversation),
        messages: messages_from_entries(&entries),
    };

    write_json(&detail)
}

fn load_conversations(
    providers: &[Box<dyn Provider>],
    settings: &HeadlessSettings,
    provider_filter: Option<ProviderFilter>,
    command_local: bool,
    command_show_deleted: bool,
) -> Result<Vec<Conversation>> {
    let local = command_local || settings.cli_local || settings.config_local;
    let show_deleted_projects = command_show_deleted || settings.show_deleted_projects;
    let mut conversations = if local {
        load_local_conversations(
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
    for (idx, conversation) in conversations.iter_mut().enumerate() {
        conversation.index = idx;
    }

    Ok(conversations)
}

fn load_global_conversations(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug: Option<DebugLevel>,
    provider_filter: Option<ProviderFilter>,
) -> Result<Vec<Conversation>> {
    let mut conversations = Vec::new();

    for provider in providers {
        if !provider_filter_matches(provider_filter, &provider.kind()) {
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

fn load_local_conversations(
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

    let mut conversations = Vec::new();
    for provider in providers {
        if !provider_filter_matches(provider_filter, &provider.kind()) {
            continue;
        }

        if let Ok(mut provider_conversations) = provider.load_conversations(show_last, debug) {
            if provider.kind() != ProviderKind::Claude {
                provider_conversations.retain(|conversation| {
                    conversation
                        .project_path
                        .as_ref()
                        .is_some_and(|path| path == &current_dir)
                });
            }
            conversations.extend(provider_conversations);
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
                        provider_key(&conversation.provider),
                        conversation.id,
                        conversation.path.display()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(AppError::CommandError(format!(
                "Ambiguous conversation target '{}'; matches: {}",
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
            provider: provider_key(&conversation.provider),
            id: conversation.id.clone(),
            path: conversation.path.to_string_lossy().to_string(),
            timestamp: conversation.timestamp.to_rfc3339(),
            project_name: conversation.project_name.clone(),
            project_path: optional_path_to_string(conversation.project_path.as_ref()),
            cwd: optional_path_to_string(conversation.cwd.as_ref()),
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
    for entry in entries {
        match entry {
            LogEntry::Summary { summary } => {
                let mut message = MessageDto::new("summary", None);
                message.text = Some(summary.clone());
                messages.push(message);
            }
            LogEntry::User {
                message,
                timestamp,
                cwd: _,
                ..
            } => push_user_message(&mut messages, message, timestamp),
            LogEntry::Assistant {
                message, timestamp, ..
            } => push_assistant_message(&mut messages, message, timestamp),
            LogEntry::Progress { data, .. } => {
                if let Some(progress) = parse_agent_progress(data) {
                    let AgentContent::Blocks(blocks) = &progress.message.message.content;
                    let role = format!("agent_{}", progress.message.message_type);
                    for block in blocks {
                        push_content_block(
                            &mut messages,
                            &role,
                            None,
                            None,
                            Some(progress.agent_id.as_str()),
                            block,
                        );
                    }
                }
            }
            LogEntry::System {
                subtype,
                level,
                duration_ms,
                ..
            } => {
                let mut message = MessageDto::new("system", None);
                message.subtype = Some(subtype.clone());
                message.text = level.clone();
                message.duration_ms = *duration_ms;
                messages.push(message);
            }
            LogEntry::FileHistorySnapshot { .. } | LogEntry::Unknown => {}
        }
    }
    messages
}

fn push_user_message(
    messages: &mut Vec<MessageDto>,
    message: &crate::claude::UserMessage,
    timestamp: &str,
) {
    match &message.content {
        UserContent::String(text) => {
            push_text_message(messages, "user", Some(timestamp), None, None, text);
        }
        UserContent::Blocks(blocks) => {
            for block in blocks {
                push_content_block(messages, "user", Some(timestamp), None, None, block);
            }
        }
    }
}

fn push_assistant_message(
    messages: &mut Vec<MessageDto>,
    message: &crate::claude::AssistantMessage,
    timestamp: &str,
) {
    for block in &message.content {
        push_content_block(
            messages,
            "assistant",
            Some(timestamp),
            message.model.as_deref(),
            None,
            block,
        );
    }
}

fn push_content_block(
    messages: &mut Vec<MessageDto>,
    text_role: &str,
    timestamp: Option<&str>,
    model: Option<&str>,
    agent_id: Option<&str>,
    block: &ContentBlock,
) {
    match block {
        ContentBlock::Text { text } => {
            push_text_message(messages, text_role, timestamp, model, agent_id, text);
        }
        ContentBlock::ToolUse { name, input, .. } => {
            let mut message = MessageDto::new("tool_call", timestamp);
            message.tool_name = Some(name.clone());
            message.tool_input = Some(input.clone());
            message.model = model.map(ToString::to_string);
            message.agent_id = agent_id.map(ToString::to_string);
            messages.push(message);
        }
        ContentBlock::ToolResult { content, .. } => {
            let mut message = MessageDto::new("tool_result", timestamp);
            message.text = tool_result_text(content.as_ref());
            message.tool_result = content.clone();
            message.agent_id = agent_id.map(ToString::to_string);
            messages.push(message);
        }
        ContentBlock::Thinking { thinking, .. } => {
            let mut message = MessageDto::new("thinking", timestamp);
            message.thinking = Some(thinking.clone());
            message.model = model.map(ToString::to_string);
            message.agent_id = agent_id.map(ToString::to_string);
            messages.push(message);
        }
        ContentBlock::Image { source } => {
            let mut message = MessageDto::new("image", timestamp);
            message.source = Some(source.clone());
            message.agent_id = agent_id.map(ToString::to_string);
            messages.push(message);
        }
    }
}

fn push_text_message(
    messages: &mut Vec<MessageDto>,
    role: &str,
    timestamp: Option<&str>,
    model: Option<&str>,
    agent_id: Option<&str>,
    text: &str,
) {
    if text.trim().is_empty() {
        return;
    }

    let mut message = MessageDto::new(role, timestamp);
    message.text = Some(text.to_string());
    message.model = model.map(ToString::to_string);
    message.agent_id = agent_id.map(ToString::to_string);
    messages.push(message);
}

fn tool_result_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(items)) => {
            let texts: Vec<_> = items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n\n"))
        }
        _ => None,
    }
}

fn write_json<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, value)?;
    writeln!(handle)?;
    Ok(())
}

fn write_jsonl<T: Serialize>(values: &[T]) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    for value in values {
        serde_json::to_writer(&mut handle, value)?;
        writeln!(handle)?;
    }
    Ok(())
}

fn provider_filter_matches(filter: Option<ProviderFilter>, kind: &ProviderKind) -> bool {
    filter.is_none_or(|filter| match filter {
        ProviderFilter::Claude => kind == &ProviderKind::Claude,
        ProviderFilter::Codex => kind == &ProviderKind::Codex,
        ProviderFilter::Cursor => kind == &ProviderKind::Cursor,
        ProviderFilter::CursorAgent => kind == &ProviderKind::CursorAgent,
    })
}

fn provider_key(kind: &ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Claude => "claude",
        ProviderKind::Codex => "codex",
        ProviderKind::Cursor => "cursor",
        ProviderKind::CursorAgent => "cursor-agent",
    }
}

fn optional_path_to_string(path: Option<&PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude::{AssistantMessage, TokenUsage, UserMessage};
    use chrono::{Local, TimeZone};

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
                    role: "user".to_string(),
                    content: UserContent::String("run tests".to_string()),
                },
                timestamp: "2026-06-19T10:00:00-07:00".to_string(),
                uuid: None,
                cwd: None,
            },
            LogEntry::Assistant {
                message: AssistantMessage {
                    role: "assistant".to_string(),
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
                            signature: "sig".to_string(),
                        },
                    ],
                    model: Some("claude-test".to_string()),
                    usage: Some(TokenUsage::default()),
                    id: None,
                },
                timestamp: "2026-06-19T10:00:01-07:00".to_string(),
                uuid: None,
            },
        ];

        let messages = messages_from_entries(&entries);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[2].role, "tool_call");
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
}
