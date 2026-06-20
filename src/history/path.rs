//! Path encoding/decoding utilities for Claude project directories.
//!
//! Claude encodes project paths as directory names by replacing non-alphanumeric
//! characters (except `-`) with `-`. This module provides utilities to convert
//! between paths and their encoded forms.

use std::path::{Path, PathBuf};

/// Lossily render an optional path as an optional string.
pub fn path_to_string(path: Option<&Path>) -> Option<String> {
    path.map(|path| path.to_string_lossy().to_string())
}

/// Convert the current working directory into Claude's project directory name.
pub fn convert_path_to_project_dir_name(path: &Path) -> String {
    path.to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Format a path into a short display name.
///
/// For worktree paths like `/Users/user/code/mnemonai__worktrees/claude-search`,
/// returns `mnemonai/claude-search` to show both the main project and worktree name.
///
/// For regular paths, returns just the folder name.
pub fn format_short_name_from_path(path: &Path) -> String {
    // If the path is the user's home directory, display as ~
    if let Some(home) = home::home_dir()
        && path == home
    {
        return "~".to_string();
    }

    let path_str = path.to_string_lossy();

    // Check for worktree pattern in the path
    if let Some(wt_pos) = path_str
        .find("__worktrees/")
        .or_else(|| path_str.find("/.worktrees/"))
    {
        let is_hidden = path_str[wt_pos..].starts_with("/.");
        let separator_len = if is_hidden {
            "/.worktrees/".len()
        } else {
            "__worktrees/".len()
        };

        // Get main project (folder before __worktrees)
        let before = &path_str[..wt_pos];
        let main_project = Path::new(before)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Get worktree name (folder after __worktrees/)
        let after = &path_str[wt_pos + separator_len..];
        let worktree = after.split('/').next().unwrap_or("");

        if !main_project.is_empty() && !worktree.is_empty() {
            return format!("{}/{}", main_project, worktree);
        }
    }

    // Not a worktree, just return the folder name
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_str.into_owned())
}

/// Whether a conversation's project directory should count as "live" for the
/// deleted-projects filter.
///
/// A path is live if it still exists. But PR-review and feature worktrees are
/// ephemeral by design — they get torn down after use — yet the conversations
/// that happened inside them are still about a repository you have. So if the
/// path looks like a worktree path and the repository it branched from still
/// exists, we treat it as live too. Only when the whole repository is gone do
/// we consider the conversation's project deleted.
pub fn project_path_is_live(path: &Path) -> bool {
    resolve_project_dir(path).is_some()
}

/// Resolve the directory a resumed session should be launched from, for a
/// conversation whose recorded project path may have been a torn-down worktree.
///
/// - The path itself if it still exists.
/// - Otherwise, for a worktree-shaped path, the repository the worktree branched
///   from, found across the two worktree topologies:
///   - **nested** (`<repo>/.worktrees/<name>`, `<repo>/.../worktrees/<name>`):
///     the repo is a surviving ancestor of the worktree;
///   - **sibling** (`<repo>__worktrees/<name>`): the repo is a sibling, reached
///     by stripping the `__worktrees` suffix from the container directory.
/// - Otherwise `None`: the whole project is gone, so there's nowhere to resume.
pub fn resolve_project_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if !is_worktree_path(path) {
        return None;
    }

    // `ancestors()` yields the path itself first; skip it since we know it's gone.
    for ancestor in path.ancestors().skip(1) {
        // Sibling layout: `…/<project>__worktrees/<name>` — the repo is a sibling
        // of the worktrees container, named by the stripped prefix.
        if let Some(stem) = ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix("__worktrees"))
        {
            let repo = ancestor.with_file_name(stem);
            if is_git_repo(&repo) {
                return Some(repo);
            }
        }
        // Nested layout: an ancestor is itself the repo.
        if is_git_repo(ancestor) {
            return Some(ancestor.to_path_buf());
        }
    }

    None
}

/// Detect the worktree-container path components: an exact `worktrees` or
/// `.worktrees`, or a `<project>__worktrees` sibling-layout directory. The
/// suffix is matched precisely so unrelated names like `client-worktrees`
/// don't trip the deleted-project rescue.
fn is_worktree_path(path: &Path) -> bool {
    path.components()
        .any(|c| is_worktree_component(c.as_os_str().to_str()))
}

fn is_worktree_component(name: Option<&str>) -> bool {
    name.is_some_and(|name| {
        let lower = name.to_ascii_lowercase();
        lower == "worktrees" || lower == ".worktrees" || lower.ends_with("__worktrees")
    })
}

/// A normal repo has a `.git` directory; a linked worktree has a `.git` file.
fn is_git_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

/// Decode a project directory name back to a path (simple heuristic fallback).
///
/// Claude's encoding replaces all non-alphanumeric characters (except `-`) with `-`.
/// This means `/`, `_`, and `.` all become `-`, making the encoding lossy.
///
/// This is only used as a fallback for old JSONL files that don't have the cwd field.
/// The cwd field from JSONL provides the accurate path and should be preferred.
pub fn decode_project_dir_name_to_path(encoded: &str) -> PathBuf {
    PathBuf::from(decode_with_double_dash_as(encoded, "__"))
}

/// Decode with a specific replacement for double dashes
fn decode_with_double_dash_as(encoded: &str, double_dash_replacement: &str) -> String {
    let mut result = String::with_capacity(encoded.len());
    let mut chars = encoded.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '-' {
            let mut count = 1;
            while chars.peek() == Some(&'-') {
                chars.next();
                count += 1;
            }

            match count {
                1 => result.push('/'),
                2 => result.push_str(double_dash_replacement),
                n => {
                    result.push('/');
                    for _ in 0..((n - 1) / 2) {
                        result.push_str(double_dash_replacement);
                    }
                    if (n - 1) % 2 == 1 {
                        result.push('/');
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Decode a project directory name back to a readable path (for display purposes).
///
/// Claude's encoding replaces all non-alphanumeric characters (except `-`) with `-`.
/// This means `/`, `_`, and `.` all become `-`, making the encoding lossy and
/// impossible to reverse perfectly.
///
/// We use a heuristic based on consecutive dash count:
/// - Odd (1, 3, 5...): `/` followed by underscores (e.g. `-` -> `/`, `---` -> `/__`)
/// - Even (2, 4...): All underscores (e.g. `--` -> `__`)
///
/// This prioritizes `__` (common in directory names like git worktrees) over `/_`.
/// Single underscores and dots in the original path will be incorrectly decoded as `/`,
/// but the result is still recognizable enough for project selection in fzf.
pub fn decode_project_dir_name(encoded: &str) -> String {
    let mut result = String::with_capacity(encoded.len());
    let mut chars = encoded.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '-' {
            // Count consecutive dashes
            let mut count = 1;
            while chars.peek() == Some(&'-') {
                chars.next();
                count += 1;
            }

            if count % 2 == 1 {
                // Odd: first is '/', rest are '_'
                result.push('/');
                for _ in 0..(count - 1) {
                    result.push('_');
                }
            } else {
                // Even: all are '_'
                for _ in 0..count {
                    result.push('_');
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Encoding tests ===

    #[test]
    fn converts_various_separators_and_punctuation() {
        let path = Path::new("/Users/user/code/workmux/.worktrees/uncommitted");
        let converted = convert_path_to_project_dir_name(path);
        assert_eq!(converted, "-Users-user-code-workmux--worktrees-uncommitted");
    }

    #[test]
    fn preserves_alphanumeric_and_existing_dashes() {
        let path = Path::new("/tmp/foo-Bar123");
        let converted = convert_path_to_project_dir_name(path);
        assert_eq!(converted, "-tmp-foo-Bar123");
    }

    #[test]
    fn encodes_worktree_with_double_underscore() {
        let path = Path::new("/Users/user/code/mnemonai__worktrees/claude-search");
        let converted = convert_path_to_project_dir_name(path);
        assert_eq!(
            converted,
            "-Users-user-code-mnemonai--worktrees-claude-search"
        );
    }

    #[test]
    fn encodes_hidden_directory() {
        let path = Path::new("/Users/user/dotfiles/.config/karabiner");
        let converted = convert_path_to_project_dir_name(path);
        assert_eq!(converted, "-Users-user-dotfiles--config-karabiner");
    }

    // === Display decode tests (decode_project_dir_name) ===

    #[test]
    fn decodes_consecutive_dashes_to_underscores() {
        // Double dash -> __ (even count = all underscores)
        let encoded = "-Users-user-code-myproject--worktrees-feature";
        let decoded = decode_project_dir_name(encoded);
        assert_eq!(decoded, "/Users/user/code/myproject__worktrees/feature");

        // Triple dash -> /__ (odd count = slash + underscores)
        let encoded = "-Users-user-code-myproject---worktrees-feature";
        let decoded = decode_project_dir_name(encoded);
        assert_eq!(decoded, "/Users/user/code/myproject/__worktrees/feature");
    }

    #[test]
    fn decodes_single_dashes_to_slashes() {
        let encoded = "-tmp-foo-Bar123";
        let decoded = decode_project_dir_name(encoded);
        assert_eq!(decoded, "/tmp/foo/Bar123");
    }

    // === Fallback decode tests (decode_with_double_dash_as) ===

    #[test]
    fn decode_with_double_dash_as_underscore() {
        let encoded = "-Users-user-code-project--worktrees-feature";
        let decoded = decode_with_double_dash_as(encoded, "__");
        assert_eq!(decoded, "/Users/user/code/project__worktrees/feature");
    }

    #[test]
    fn decode_with_double_dash_as_hidden_dir() {
        let encoded = "-Users-user-dotfiles--config-karabiner";
        let decoded = decode_with_double_dash_as(encoded, "/.");
        assert_eq!(decoded, "/Users/user/dotfiles/.config/karabiner");
    }

    #[test]
    fn decode_preserves_dashes_in_folder_names_in_fallback() {
        // Note: The fallback decode can't distinguish dashes in folder names
        // from path separators - this is expected behavior
        let encoded = "-Users-user-code-my-project";
        let decoded = decode_with_double_dash_as(encoded, "__");
        // This incorrectly decodes to /Users/user/code/my/project
        // because single dashes are treated as path separators
        assert_eq!(decoded, "/Users/user/code/my/project");
    }

    // === Worktree path structure tests ===

    #[test]
    fn worktree_encoded_pattern() {
        // Verify the encoding pattern for worktrees
        let path = Path::new("/Users/user/code/WalkingMate__worktrees/template-engine");
        let encoded = convert_path_to_project_dir_name(path);
        assert_eq!(
            encoded,
            "-Users-user-code-WalkingMate--worktrees-template-engine"
        );

        // The --worktrees- pattern should be detectable
        assert!(encoded.contains("--worktrees-"));
    }

    #[test]
    fn extract_worktree_name_from_encoded() {
        let encoded = "-Users-user-code-WalkingMate--worktrees-template-engine";

        // Find the worktree marker
        let wt_pos = encoded.find("--worktrees-").unwrap();

        // Extract worktree name (everything after --worktrees-)
        let worktree_name = &encoded[wt_pos + "--worktrees-".len()..];
        assert_eq!(worktree_name, "template-engine");
    }

    #[test]
    fn extract_project_name_before_worktrees() {
        let encoded = "-Users-user-code-WalkingMate--worktrees-template-engine";

        // Find the worktree marker
        let wt_pos = encoded.find("--worktrees-").unwrap();

        // Extract the part before --worktrees
        let before_wt = &encoded[..wt_pos];
        assert_eq!(before_wt, "-Users-user-code-WalkingMate");

        // When decoded with filesystem check, this should give us WalkingMate as the project name
        // For fallback, it decodes to a path ending in WalkingMate
        let decoded = decode_with_double_dash_as(before_wt, "__");
        assert_eq!(decoded, "/Users/user/code/WalkingMate");
    }

    // === format_project_short_name tests (worktree display) ===

    #[test]
    fn format_short_name_extracts_worktree_pattern() {
        // Test the worktree pattern detection in decoded paths
        let path = "/Users/user/code/WalkingMate__worktrees/template-engine";

        // Check for worktree pattern
        assert!(path.contains("__worktrees/"));

        // Extract main project
        let wt_pos = path.find("__worktrees/").unwrap();
        let before = &path[..wt_pos];
        let main_project = Path::new(before)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(main_project, "WalkingMate");

        // Extract worktree name
        let after = &path[wt_pos + "__worktrees/".len()..];
        let worktree = after.split('/').next().unwrap();
        assert_eq!(worktree, "template-engine");

        // Combined display
        let display = format!("{}/{}", main_project, worktree);
        assert_eq!(display, "WalkingMate/template-engine");
    }

    // === project_path_is_live tests ===

    #[test]
    fn is_worktree_path_matches_common_shapes() {
        assert!(is_worktree_path(Path::new(
            "/Users/u/code/mnemonai__worktrees/feature"
        )));
        assert!(is_worktree_path(Path::new(
            "/Users/u/code/workmux/.worktrees/uncommitted"
        )));
        assert!(is_worktree_path(Path::new(
            "/Users/u/code/repo/.agent-pr-review/worktrees/pr-4"
        )));
        assert!(!is_worktree_path(Path::new("/Users/u/code/repo/src")));
        // A name that merely ends with "worktrees" must not be treated as one.
        assert!(!is_worktree_path(Path::new(
            "/Users/u/code/client-worktrees/thing"
        )));
    }

    #[test]
    fn resolve_project_dir_handles_sibling_worktrees_layout() {
        // Sibling layout: repo at `…/code/proj`, worktrees at `…/code/proj__worktrees/*`.
        let base = std::env::temp_dir().join("mnemonai_resolve_sibling");
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("proj");
        std::fs::create_dir_all(repo.join(".git")).unwrap();

        let gone = base.join("proj__worktrees/feature");
        assert!(!gone.exists());
        assert_eq!(resolve_project_dir(&gone), Some(repo.clone()));

        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn resolve_project_dir_returns_repo_for_gone_worktree() {
        let repo = std::env::temp_dir().join("mnemonai_resolve_repo");
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(repo.join(".git")).unwrap();

        // A gone worktree resolves to the surviving repo it branched from.
        let gone = repo.join(".agent-pr-review/worktrees/pr-4");
        assert_eq!(resolve_project_dir(&gone), Some(repo.clone()));
        // An existing directory resolves to itself.
        assert_eq!(resolve_project_dir(&repo), Some(repo.clone()));
        // A gone, non-worktree path has nowhere to resume.
        assert_eq!(resolve_project_dir(&repo.join("src/gone")), None);

        std::fs::remove_dir_all(&repo).unwrap();
    }

    #[test]
    fn live_when_path_exists() {
        // The repo we're running in always exists.
        let cwd = std::env::current_dir().unwrap();
        assert!(project_path_is_live(&cwd));
    }

    #[test]
    fn not_live_when_nonexistent_and_not_a_worktree() {
        let path = Path::new("/Users/u/code/this-project-was-deleted-xyz");
        assert!(!project_path_is_live(path));
    }

    #[test]
    fn live_when_deleted_worktree_of_surviving_repo() {
        // Build a temp git repo, then a worktree-shaped subpath that doesn't
        // exist on disk. The repo survives, so the conversation stays live.
        let repo = std::env::temp_dir().join("mnemonai_wt_test_repo");
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(repo.join(".git")).unwrap();

        let gone_worktree = repo.join(".agent-pr-review/worktrees/pr-4");
        assert!(!gone_worktree.exists());
        assert!(project_path_is_live(&gone_worktree));

        // But a deleted worktree of a *deleted* repo is not rescued.
        let orphan =
            std::env::temp_dir().join("mnemonai_no_such_repo_xyz/.agent-pr-review/worktrees/pr-4");
        assert!(!project_path_is_live(&orphan));

        std::fs::remove_dir_all(&repo).unwrap();
    }

    #[test]
    fn format_short_name_hidden_worktrees() {
        // Test .worktrees pattern (hidden worktrees folder)
        let path = "/Users/user/code/workmux/.worktrees/uncommitted";

        // Check for hidden worktree pattern
        assert!(path.contains("/.worktrees/"));

        let wt_pos = path.find("/.worktrees/").unwrap();
        let before = &path[..wt_pos];
        let main_project = Path::new(before)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(main_project, "workmux");

        let after = &path[wt_pos + "/.worktrees/".len()..];
        let worktree = after.split('/').next().unwrap();
        assert_eq!(worktree, "uncommitted");
    }
}
