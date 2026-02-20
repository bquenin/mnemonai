//! Syntax highlighting for code blocks using syntect.

use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME_SET: OnceLock<ThemeSet> = OnceLock::new();

fn get_syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn get_theme_set() -> &'static ThemeSet {
    THEME_SET.get_or_init(ThemeSet::load_defaults)
}

/// Normalize common language aliases to their canonical names
fn normalize_language(lang: &str) -> &str {
    // Take only the first token (handle "rust,ignore" or "rust title=x")
    let lang = lang.split([',', ' ']).next().unwrap_or(lang).trim();

    match lang.to_lowercase().as_str() {
        "js" => "javascript",
        "ts" => "typescript",
        "sh" | "shell" => "bash",
        "yml" => "yaml",
        "py" => "python",
        "rb" => "ruby",
        "md" => "markdown",
        "dockerfile" => "Dockerfile",
        _ => lang,
    }
}

/// Highlighted token with styling information
pub struct HighlightedToken {
    pub text: String,
    pub fg: (u8, u8, u8),
    pub bold: bool,
    pub italic: bool,
}

/// Highlight code and return ANSI-escaped string for CLI output.
/// Returns None if colors are disabled or language is unknown.
pub fn highlight_code_ansi(code: &str, lang: &str) -> Option<String> {
    // Respect NO_COLOR and color-disable settings
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        return None;
    }

    let ps = get_syntax_set();
    let ts = get_theme_set();

    let lang = normalize_language(lang);
    let syntax = ps.find_syntax_by_token(lang)?;
    let theme = ts.themes.get("base16-ocean.dark")?;

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut output = String::new();

    for line in LinesWithEndings::from(code) {
        let ranges = highlighter.highlight_line(line, ps).ok()?;
        for (style, text) in ranges {
            let fg = style.foreground;
            // Build ANSI escape sequence for foreground color
            output.push_str(&format!("\x1b[38;2;{};{};{}m", fg.r, fg.g, fg.b));
            if style.font_style.contains(FontStyle::BOLD) {
                output.push_str("\x1b[1m");
            }
            if style.font_style.contains(FontStyle::ITALIC) {
                output.push_str("\x1b[3m");
            }
            output.push_str(text);
            output.push_str("\x1b[0m");
        }
    }

    Some(output)
}

/// Highlight code and return styled tokens for TUI rendering.
/// Returns None if language is unknown.
pub fn highlight_code_tui(code: &str, lang: &str) -> Option<Vec<Vec<HighlightedToken>>> {
    let ps = get_syntax_set();
    let ts = get_theme_set();

    let lang = normalize_language(lang);
    let syntax = ps.find_syntax_by_token(lang)?;
    let theme = ts.themes.get("base16-ocean.dark")?;

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines = Vec::new();

    for line in LinesWithEndings::from(code) {
        let ranges = highlighter.highlight_line(line, ps).ok()?;
        let tokens: Vec<HighlightedToken> = ranges
            .into_iter()
            .map(|(style, text)| HighlightedToken {
                text: text.to_string(),
                fg: (style.foreground.r, style.foreground.g, style.foreground.b),
                bold: style.font_style.contains(FontStyle::BOLD),
                italic: style.font_style.contains(FontStyle::ITALIC),
            })
            .collect();
        lines.push(tokens);
    }

    Some(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_language() {
        assert_eq!(normalize_language("js"), "javascript");
        assert_eq!(normalize_language("ts"), "typescript");
        assert_eq!(normalize_language("sh"), "bash");
        assert_eq!(normalize_language("rust"), "rust");
        assert_eq!(normalize_language("rust,ignore"), "rust");
        assert_eq!(normalize_language("rust title=x"), "rust");
    }

    #[test]
    fn test_highlight_known_language() {
        // Force colors for test
        colored::control::set_override(true);

        let code = "let x = 1;";
        let result = highlight_code_ansi(code, "rust");
        assert!(result.is_some());
        let output = result.unwrap();
        // Should contain ANSI escape codes
        assert!(
            output.contains("\x1b["),
            "Expected ANSI codes, got: {:?}",
            output
        );
        // Verify multiple color codes (syntax highlighting produces different colors)
        assert!(
            output.matches("\x1b[38;2;").count() > 1,
            "Expected multiple color codes for different tokens"
        );

        colored::control::unset_override();
    }

    #[test]
    fn test_highlight_unknown_language() {
        let code = "some code";
        let result = highlight_code_ansi(code, "unknown_language_xyz");
        // Should return None for unknown languages
        assert!(result.is_none());
    }

    #[test]
    fn test_highlight_tui_known_language() {
        let code = "fn main() {}";
        let result = highlight_code_tui(code, "rust");
        assert!(result.is_some());
        let lines = result.unwrap();
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_highlight_with_alias() {
        // Force colors for test
        colored::control::set_override(true);

        let code = "const x = 1;";
        let result = highlight_code_ansi(code, "js");
        assert!(result.is_some());

        colored::control::unset_override();
    }
}
