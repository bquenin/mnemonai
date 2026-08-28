/// How a user-role entry's text should be attributed.
///
/// Transcripts record several kinds of content under `type: "user"`: what the
/// user actually said or typed, output produced by commands they ran (shell
/// commands via the `!` prefix, slash-command stdout), and internal markers.
/// Only the first is user speech.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UserText {
    /// Genuine user speech (possibly cleaned of wrapper tags).
    Speech(String),
    /// Command/shell output carried in a user-role entry — not user speech.
    Output(String),
    /// Internal marker content that should not be displayed at all.
    Skip,
}

/// Classify user message text, stripping the XML wrappers transcripts use and
/// separating genuine user speech from command output.
pub fn classify_user_text(text: &str) -> UserText {
    let trimmed = text.trim();

    // Check for local-command-caveat - skip these system messages entirely
    if trimmed.starts_with("<local-command-caveat>") && trimmed.ends_with("</local-command-caveat>")
    {
        return UserText::Skip;
    }

    // local-command-stdout carries the output of a slash command, not user speech
    if trimmed.starts_with("<local-command-stdout>") && trimmed.ends_with("</local-command-stdout>")
    {
        let tag_start = "<local-command-stdout>".len();
        let tag_end = trimmed.len() - "</local-command-stdout>".len();
        let inner = trimmed[tag_start..tag_end].trim();
        if inner.is_empty() {
            return UserText::Skip;
        }
        return UserText::Output(inner.to_string());
    }

    // `! <cmd>` shell commands: the user typed the command, so it is speech.
    // Mirror how Claude Code displays it by re-adding the `! ` prefix.
    if let Some(command) = extract_xml_tag_content(trimmed, "bash-input") {
        return UserText::Speech(format!("! {}", command.trim()));
    }

    // ...and the paired stdout/stderr entry is the command's output, not speech
    if trimmed.contains("<bash-stdout>") || trimmed.contains("<bash-stderr>") {
        let stdout = extract_xml_tag_content(trimmed, "bash-stdout")
            .unwrap_or("")
            .trim();
        let stderr = extract_xml_tag_content(trimmed, "bash-stderr")
            .unwrap_or("")
            .trim();
        let parts: Vec<&str> = [stdout, stderr]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return UserText::Skip;
        }
        return UserText::Output(parts.join("\n"));
    }

    // Check if this is a command message with <command-name> tag
    if let Some(start) = trimmed.find("<command-name>")
        && let Some(end) = trimmed.find("</command-name>")
    {
        let content_start = start + "<command-name>".len();
        if content_start < end {
            let command_name = &trimmed[content_start..end];

            // Skip /clear commands - internal context-clearing, not meaningful to display
            if command_name == "/clear" {
                return UserText::Skip;
            }

            // Also extract command args if present
            if let Some(args_start) = trimmed.find("<command-args>")
                && let Some(args_end) = trimmed.find("</command-args>")
            {
                let args_content_start = args_start + "<command-args>".len();
                if args_content_start < args_end {
                    let args = trimmed[args_content_start..args_end].trim();
                    if !args.is_empty() {
                        return UserText::Speech(format!("{} {}", command_name, args));
                    }
                }
            }

            return UserText::Speech(command_name.to_string());
        }
    }

    // Handle Cursor Agent CLI user messages wrapped in <user_query> tags,
    // optionally preceded by a <timestamp> tag
    if trimmed.contains("<user_query>") {
        let text_to_parse = strip_xml_tag(trimmed, "timestamp");
        if let Some(inner) = extract_xml_tag_content(&text_to_parse, "user_query") {
            let inner = inner.trim();
            if inner.is_empty() {
                return UserText::Skip;
            }
            return UserText::Speech(inner.to_string());
        }
    }

    // Return original text for non-command messages
    UserText::Speech(text.to_string())
}

fn extract_xml_tag_content<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = text.find(&open)?;
    let content_start = start + open.len();
    let end = text[content_start..].find(&close)?;
    Some(&text[content_start..content_start + end])
}

fn strip_xml_tag(text: &str, tag: &str) -> String {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    if let Some(start) = text.find(&open)
        && let Some(close_start) = text.find(&close)
    {
        let after = close_start + close.len();
        let mut result = String::with_capacity(text.len());
        result.push_str(&text[..start]);
        result.push_str(&text[after..]);
        result
    } else {
        text.to_string()
    }
}

pub fn short_agent_id(agent_id: &str) -> &str {
    &agent_id[..agent_id.len().min(7)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech(text: &str) -> UserText {
        UserText::Speech(text.to_string())
    }

    fn output(text: &str) -> UserText {
        UserText::Output(text.to_string())
    }

    #[test]
    fn skips_local_command_caveat() {
        let caveat = "<local-command-caveat>Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.</local-command-caveat>";
        assert_eq!(classify_user_text(caveat), UserText::Skip);
    }

    #[test]
    fn skips_local_command_caveat_with_whitespace() {
        let caveat = "  <local-command-caveat>Some caveat text</local-command-caveat>  ";
        assert_eq!(classify_user_text(caveat), UserText::Skip);
    }

    #[test]
    fn preserves_normal_text() {
        assert_eq!(classify_user_text("Hello world"), speech("Hello world"));
    }

    #[test]
    fn skips_empty_stdout() {
        assert_eq!(
            classify_user_text("<local-command-stdout></local-command-stdout>"),
            UserText::Skip
        );
        assert_eq!(
            classify_user_text("<local-command-stdout>   </local-command-stdout>"),
            UserText::Skip
        );
    }

    /// Slash-command stdout is the command's output, not something the user said.
    #[test]
    fn nonempty_stdout_is_output() {
        assert_eq!(
            classify_user_text("<local-command-stdout>output here</local-command-stdout>"),
            output("output here")
        );
    }

    /// The user typed the `!` command, so it is speech — displayed the way
    /// Claude Code shows it, with a leading `! `.
    #[test]
    fn bash_input_is_speech_with_bang_prefix() {
        assert_eq!(
            classify_user_text("<bash-input>gh auth refresh -h github.com</bash-input>"),
            speech("! gh auth refresh -h github.com")
        );
    }

    #[test]
    fn bash_stdout_is_output() {
        assert_eq!(
            classify_user_text(
                "<bash-stdout>! First copy your one-time code\ndone</bash-stdout><bash-stderr></bash-stderr>"
            ),
            output("! First copy your one-time code\ndone")
        );
    }

    #[test]
    fn skips_empty_bash_output() {
        assert_eq!(
            classify_user_text("<bash-stdout></bash-stdout><bash-stderr>  </bash-stderr>"),
            UserText::Skip
        );
    }

    #[test]
    fn bash_stderr_only_is_output() {
        assert_eq!(
            classify_user_text("<bash-stdout></bash-stdout><bash-stderr>boom</bash-stderr>"),
            output("boom")
        );
    }

    #[test]
    fn bash_stdout_and_stderr_are_joined() {
        assert_eq!(
            classify_user_text("<bash-stdout>out</bash-stdout><bash-stderr>err</bash-stderr>"),
            output("out\nerr")
        );
    }

    #[test]
    fn skips_clear_command() {
        assert_eq!(
            classify_user_text("<command-name>/clear</command-name>"),
            UserText::Skip
        );
        assert_eq!(
            classify_user_text(
                "<command-name>/clear</command-name>\n<command-message>clear</command-message>\n<command-args></command-args>"
            ),
            UserText::Skip
        );
    }

    #[test]
    fn extracts_other_command_names() {
        assert_eq!(
            classify_user_text("<command-name>/help</command-name>"),
            speech("/help")
        );
    }

    #[test]
    fn extracts_command_name_with_args() {
        assert_eq!(
            classify_user_text(
                "<command-name>/review</command-name>\n<command-args>high</command-args>"
            ),
            speech("/review high")
        );
    }

    #[test]
    fn strips_user_query_tags() {
        assert_eq!(
            classify_user_text("<user_query>\nperfect thanks\n</user_query>"),
            speech("perfect thanks")
        );
    }

    #[test]
    fn strips_timestamp_and_user_query() {
        let text = "<timestamp>Thursday, May 7, 2026, 10:12 PM (UTC-7)</timestamp>\n<user_query>\nperfect thanks\n</user_query>";
        assert_eq!(classify_user_text(text), speech("perfect thanks"));
    }

    #[test]
    fn skips_empty_user_query() {
        assert_eq!(
            classify_user_text("<user_query>\n\n</user_query>"),
            UserText::Skip
        );
    }

    #[test]
    fn preserves_multiline_user_query() {
        let text = "<timestamp>Tuesday, Apr 28, 2026, 3:31 PM (UTC-7)</timestamp>\n<user_query>\nRun `cat /etc/services` and `seq 1 1000` exactly once each.\nDo not run any other commands.\n</user_query>";
        let UserText::Speech(result) = classify_user_text(text) else {
            panic!("expected speech");
        };
        assert!(result.contains("Run `cat /etc/services`"));
        assert!(result.contains("Do not run any other commands."));
        assert!(!result.contains("<user_query>"));
        assert!(!result.contains("<timestamp>"));
    }
}
