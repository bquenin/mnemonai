use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum LogEntry {
    Summary {
        summary: String,
    },
    User {
        message: UserMessage,
        /// ISO 8601 timestamp when this message was sent
        timestamp: String,
        /// The working directory when this message was sent
        cwd: Option<String>,
    },
    Assistant {
        message: AssistantMessage,
        /// ISO 8601 timestamp when this message was sent
        timestamp: String,
    },
    #[serde(rename = "file-history-snapshot")]
    FileHistorySnapshot,
    Progress {
        data: serde_json::Value,
    },
    System {
        subtype: String,
        level: Option<String>,
        /// Duration in milliseconds for turn_duration entries
        #[serde(rename = "durationMs")]
        duration_ms: Option<u64>,
    },
    /// Entry types this version doesn't understand (attachment, ai-title,
    /// queue-operation, ...). Newer Claude Code builds keep introducing types;
    /// treating them as parse errors used to record megabytes of line content
    /// and context per file.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Clone)]
pub struct UserMessage {
    pub content: UserContent,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum UserContent {
    String(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, Clone)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub model: Option<String>,
    pub usage: Option<TokenUsage>,
    /// Unique message ID to deduplicate streaming entries
    pub id: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TokenUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<serde_json::Value>, // Optional in some user tool result entries
        #[serde(default)]
        is_error: Option<bool>,
        #[serde(default)]
        status: Option<String>,
    },
    Thinking {
        thinking: String,
    },
    Image {
        source: serde_json::Value,
    },
}

/// Extract text from content blocks, used for both user and assistant messages
pub fn extract_text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn extract_text_from_user(message: &UserMessage) -> String {
    match &message.content {
        UserContent::String(text) => text.clone(),
        UserContent::Blocks(blocks) => extract_text_from_blocks(blocks),
    }
}

pub fn extract_text_from_assistant(message: &AssistantMessage) -> String {
    extract_text_from_blocks(&message.content)
}

/// Extract text content from a tool result block.
///
/// Returns `Some(text)` when the content is a string or an array of text
/// blocks (e.g. `[{type: "text", text: "..."}]`), and `None` for JSON
/// structures that should be pretty-printed instead.
pub fn extract_tool_result_text(content: Option<&serde_json::Value>) -> Option<String> {
    match content {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(arr)) => {
            let texts: Vec<&str> = arr
                .iter()
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n\n"))
        }
        _ => None,
    }
}

/// Agent progress data from subagent conversations
#[derive(Debug, Deserialize, Clone)]
pub struct AgentProgressData {
    #[serde(rename = "agentId")]
    pub agent_id: String,
    pub message: AgentMessage,
}

/// Individual message within an agent conversation
#[derive(Debug, Deserialize, Clone)]
pub struct AgentMessage {
    #[serde(rename = "type")]
    pub message_type: String, // "user" or "assistant"
    pub message: AgentMessageContent,
}

/// Content of an agent message (mirrors UserMessage/AssistantMessage structure)
#[derive(Debug, Deserialize, Clone)]
pub struct AgentMessageContent {
    pub content: AgentContent,
}

/// Agent message content is always an array of content blocks
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum AgentContent {
    Blocks(Vec<ContentBlock>),
}

/// Attempt to parse agent progress data from a Progress entry
pub fn parse_agent_progress(data: &serde_json::Value) -> Option<AgentProgressData> {
    // Check if this is an agent_progress type
    if data.get("type").and_then(|t| t.as_str()) != Some("agent_progress") {
        return None;
    }
    serde_json::from_value(data.clone()).ok()
}
