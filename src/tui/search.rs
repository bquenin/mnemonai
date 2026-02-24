use crate::history::Conversation;
use chrono::{DateTime, Duration, Local};
use rayon::prelude::*;

/// Precomputed search data for a conversation
pub struct SearchableConversation {
    /// Lowercased full text for searching
    pub text_lower: String,
    /// Original full text (moved from Conversation to avoid duplication)
    pub full_text: String,
    /// Original conversation index
    pub index: usize,
}

/// Normalize text for search: lowercase only.
/// Query terms are split on whitespace and matched as substrings,
/// so URLs, paths, and other structured strings stay intact as single terms.
pub fn normalize_for_search(text: &str) -> String {
    text.to_lowercase()
}

/// Check if a character is a word separator for search purposes (Ctrl+W, highlighting).
pub fn is_word_separator(c: char) -> bool {
    c.is_whitespace()
}

/// Precompute lowercased search text for all conversations.
/// Moves `full_text` ownership from each Conversation into SearchableConversation
/// to avoid storing the same text twice in memory.
pub fn precompute_search_text(conversations: &mut [Conversation]) -> Vec<SearchableConversation> {
    conversations
        .par_iter_mut()
        .enumerate()
        .map(|(idx, conv)| {
            let full_text = std::mem::take(&mut conv.full_text);
            let text_lower = normalize_for_search(&full_text);
            SearchableConversation {
                text_lower,
                full_text,
                index: idx,
            }
        })
        .collect()
}

/// Filter and score conversations based on query.
/// Returns indices into the original conversations vec, sorted by score descending.
/// When `narrow_from` is provided, only scores those indices (used when query extends previous).
pub fn search(
    conversations: &[Conversation],
    searchable: &[SearchableConversation],
    query: &str,
    now: DateTime<Local>,
    narrow_from: Option<&[usize]>,
) -> Vec<usize> {
    let query = query.trim();
    if query.is_empty() {
        // Return all indices sorted by timestamp (already sorted in history.rs)
        return (0..conversations.len()).collect();
    }

    let query_lower = normalize_for_search(query);

    // Score conversations in parallel, optionally narrowing to a subset
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
    if query_terms.is_empty() {
        return (0..conversations.len()).collect();
    }

    let mut scored: Vec<(usize, f64, DateTime<Local>)> = if let Some(indices) = narrow_from {
        let allowed: std::collections::HashSet<usize> = indices.iter().copied().collect();
        searchable
            .par_iter()
            .filter(|s| allowed.contains(&s.index))
            .filter_map(|s| {
                let score = score_text(
                    &s.text_lower,
                    &query_terms,
                    conversations[s.index].timestamp,
                    now,
                );
                if score > 0.0 {
                    Some((s.index, score, conversations[s.index].timestamp))
                } else {
                    None
                }
            })
            .collect()
    } else {
        searchable
            .par_iter()
            .filter_map(|s| {
                let score = score_text(
                    &s.text_lower,
                    &query_terms,
                    conversations[s.index].timestamp,
                    now,
                );
                if score > 0.0 {
                    Some((s.index, score, conversations[s.index].timestamp))
                } else {
                    None
                }
            })
            .collect()
    };

    // Sort by score descending, then by timestamp descending for stability
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.cmp(&a.2))
    });

    scored.into_iter().map(|(idx, _, _)| idx).collect()
}

/// Score a conversation based on substring matching and recency.
/// Each query term (split on whitespace) must appear as a substring in the text (AND logic).
/// This preserves URLs, paths, and other structured strings as single terms.
fn score_text(
    text_lower: &str,
    query_terms: &[&str],
    timestamp: DateTime<Local>,
    now: DateTime<Local>,
) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }

    // All terms must appear as substrings (AND logic)
    for &term in query_terms {
        if !text_lower.contains(term) {
            return 0.0;
        }
    }

    (query_terms.len() as f64) * recency_multiplier(timestamp, now)
}

/// Calculate recency multiplier based on age
fn recency_multiplier(timestamp: DateTime<Local>, now: DateTime<Local>) -> f64 {
    let age = now.signed_duration_since(timestamp);

    // Handle future timestamps (shouldn't happen, but be safe)
    if age < Duration::zero() {
        return 3.0;
    }

    if age < Duration::days(1) {
        3.0
    } else if age < Duration::days(7) {
        2.0
    } else if age < Duration::days(30) {
        1.5
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Conversation;
    use std::path::PathBuf;

    fn make_conv(text: &str, timestamp: DateTime<Local>) -> Conversation {
        Conversation {
            path: PathBuf::new(),
            index: 0,
            provider: crate::history::ProviderKind::Claude,
            id: String::new(),
            timestamp,
            preview: text.to_string(),
            full_text: text.to_string(),
            project_name: None,
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: vec![],
            summary: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    #[test]
    fn search_matches_case_insensitive() {
        let now = Local::now();
        let mut convs = vec![make_conv("Hardened Runtime enabled", now)];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "harden runtime", now, None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_substring_matches() {
        let now = Local::now();
        let mut convs = vec![make_conv("hardened security", now)];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "harden", now, None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_requires_all_terms() {
        let now = Local::now();
        let mut convs = vec![make_conv("hardened security", now)];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "harden runtime", now, None);
        assert_eq!(results.len(), 0); // "runtime" not present
    }

    #[test]
    fn search_underscore_preserved_in_query() {
        let now = Local::now();
        let mut convs = vec![
            make_conv("HARDENED_RUNTIME config", now),
            make_conv("hardened runtime enabled", now),
        ];
        let searchable = precompute_search_text(&mut convs);
        // Underscore in query matches underscore in text, not space-separated
        let results = search(&convs, &searchable, "hardened_runtime", now, None);
        assert_eq!(results.len(), 1, "should only match the underscore variant");
    }

    #[test]
    fn recency_today_gets_highest_multiplier() {
        let now = Local::now();
        let timestamp = now - Duration::hours(1);
        assert_eq!(recency_multiplier(timestamp, now), 3.0);
    }

    #[test]
    fn recency_this_week_gets_medium_multiplier() {
        let now = Local::now();
        let timestamp = now - Duration::days(3);
        assert_eq!(recency_multiplier(timestamp, now), 2.0);
    }

    #[test]
    fn recency_this_month_gets_low_multiplier() {
        let now = Local::now();
        let timestamp = now - Duration::days(15);
        assert_eq!(recency_multiplier(timestamp, now), 1.5);
    }

    #[test]
    fn recency_older_gets_base_multiplier() {
        let now = Local::now();
        let timestamp = now - Duration::days(60);
        assert_eq!(recency_multiplier(timestamp, now), 1.0);
    }

    #[test]
    fn search_matches_through_punctuation() {
        let now = Local::now();
        // Text contains "#555" (e.g., a GitHub issue reference)
        let mut convs = vec![make_conv("fix issue #555 in the parser", now)];
        let searchable = precompute_search_text(&mut convs);
        // Searching for "555" (without #) should still match
        let results = search(&convs, &searchable, "555", now, None);
        assert_eq!(results.len(), 1, "555 should match #555");
        // Searching for "#555" should also match
        let results = search(&convs, &searchable, "#555", now, None);
        assert_eq!(results.len(), 1, "#555 should also match");
    }

    #[test]
    fn search_matches_path_components() {
        let now = Local::now();
        let mut convs = vec![make_conv("edit src/main.rs file", now)];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "main", now, None);
        assert_eq!(results.len(), 1, "main should match src/main.rs");
    }

    #[test]
    fn future_timestamp_gets_highest_multiplier() {
        let now = Local::now();
        let timestamp = now + Duration::hours(1);
        assert_eq!(recency_multiplier(timestamp, now), 3.0);
    }

    #[test]
    fn search_url_matches_exactly() {
        let now = Local::now();
        let mut convs = vec![
            make_conv(
                "review this PR https://github.com/org/repo/pull/1 please",
                now - Duration::days(5),
            ),
            // Different PR number — should NOT match
            make_conv(
                "check https://github.com/org/repo/pull/3 and also item 1",
                now,
            ),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(
            &convs,
            &searchable,
            "https://github.com/org/repo/pull/1",
            now,
            None,
        );
        // Only the conversation with the exact URL should match
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0);
    }
}
