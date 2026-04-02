use crate::history::Conversation;
use chrono::{DateTime, Duration, Local};
use rayon::prelude::*;

/// Size of the "topic window" — the first ~2000 characters of a conversation,
/// covering the initial exchanges where the user establishes intent.
const TOPIC_WINDOW_SIZE: usize = 2000;

/// How much extra weight topic-window matches get over body matches.
const TOPIC_WEIGHT: f64 = 3.0;

/// Precomputed search data for a conversation
pub struct SearchableConversation {
    /// Lowercased full text for searching
    pub text_lower: String,
    /// Original full text (moved from Conversation to avoid duplication)
    pub full_text: String,
    /// Byte offset where the topic window ends in `text_lower`.
    /// Matches within `text_lower[..topic_end]` are weighted higher.
    pub topic_end: usize,
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
            // Find a char-boundary at or after TOPIC_WINDOW_SIZE bytes
            let topic_end = if text_lower.len() <= TOPIC_WINDOW_SIZE {
                text_lower.len()
            } else {
                // Advance past TOPIC_WINDOW_SIZE to the next char boundary
                let mut end = TOPIC_WINDOW_SIZE;
                while !text_lower.is_char_boundary(end) && end < text_lower.len() {
                    end += 1;
                }
                end
            };
            SearchableConversation {
                text_lower,
                full_text,
                topic_end,
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
                    s.topic_end,
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
                    s.topic_end,
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

/// Count non-overlapping occurrences of `needle` in `haystack`.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

/// Score a conversation based on substring matching, term frequency density, and recency.
/// Each query term (split on whitespace) must appear as a substring in the text (AND logic).
/// This preserves URLs, paths, and other structured strings as single terms.
///
/// Scoring formula:
///   For each term:
///     weighted_count = topic_hits * TOPIC_WEIGHT + body_hits
///     per-term score = sqrt(weighted_count)
///   density = relevance / ln(text_length)
///   final = density * recency_multiplier
fn score_text(
    text_lower: &str,
    topic_end: usize,
    query_terms: &[&str],
    timestamp: DateTime<Local>,
    now: DateTime<Local>,
) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let topic_window = &text_lower[..topic_end];
    let body = &text_lower[topic_end..];

    let mut relevance = 0.0;
    for &term in query_terms {
        let topic_hits = count_occurrences(topic_window, term);
        let body_hits = count_occurrences(body, term);
        let total_hits = topic_hits + body_hits;
        if total_hits == 0 {
            return 0.0; // AND logic: all terms must be present
        }
        // Weight topic-window hits higher: early exchanges signal what the convo is about
        let weighted = (topic_hits as f64) * TOPIC_WEIGHT + (body_hits as f64);
        // sqrt dampens tool-output spam while preserving meaningful differences:
        // 1 hit = 1.0, 5 hits ≈ 2.2, 20 hits ≈ 4.5, 100 hits = 10.0
        relevance += weighted.sqrt();
    }

    // Normalize by text length so dense matches in short conversations rank higher.
    // Use ln(len) to soften the penalty — a conversation twice as long isn't half as relevant.
    // Floor at ln(500) ≈ 6.2 so very short texts don't get an outsized boost.
    let len_norm = (text_lower.len().max(500) as f64).ln();
    let density = relevance / len_norm;

    density * recency_multiplier(timestamp, now)
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

    #[test]
    fn dense_match_ranks_higher_than_sparse() {
        let now = Local::now();
        // Same timestamp so recency doesn't influence ranking
        let mut convs = vec![
            // Sparse: "deploy" appears once in a long text
            make_conv(
                &format!("we need to deploy the app {}", "blah ".repeat(200)),
                now,
            ),
            // Dense: "deploy" appears many times in a short text
            make_conv(
                "deploy deploy deploy the deploy fix for deploy",
                now,
            ),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "deploy", now, None);
        assert_eq!(results.len(), 2);
        // Dense conversation should rank first
        assert_eq!(results[0], 1, "dense match should rank higher than sparse");
    }

    #[test]
    fn highly_relevant_old_convo_beats_barely_relevant_recent() {
        let now = Local::now();
        let padding = "unrelated stuff ".repeat(200); // push the mention past topic window
        let mut convs = vec![
            // Old but very relevant: "webpack" mentioned many times in topic window
            make_conv(
                "webpack config webpack loader webpack plugin webpack bundle webpack optimization",
                now - Duration::days(60),
            ),
            // Recent but barely relevant: "webpack" mentioned once, buried past the topic window
            make_conv(
                &format!("{padding} someone mentioned webpack once"),
                now,
            ),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "webpack", now, None);
        assert_eq!(results.len(), 2);
        // The highly relevant old conversation should beat the barely-relevant recent one
        assert_eq!(results[0], 0, "highly relevant old convo should rank above barely relevant recent one");
    }

    #[test]
    fn topic_window_match_ranks_higher_than_body_match() {
        let now = Local::now();
        // Build two conversations with the same total number of "migrate" hits,
        // but one has hits in the topic window (first ~2000 chars) and the other in the body.
        let padding = "x ".repeat(1500); // ~3000 chars of padding
        let mut convs = vec![
            // "migrate" only in the body (after the topic window)
            make_conv(
                &format!("{padding}migrate migrate migrate"),
                now,
            ),
            // "migrate" in the topic window (beginning of conversation)
            make_conv(
                &format!("migrate migrate migrate {padding}"),
                now,
            ),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "migrate", now, None);
        assert_eq!(results.len(), 2);
        // The conversation with topic-window hits should rank first
        assert_eq!(results[0], 1, "topic-window match should rank higher than body-only match");
    }

    #[test]
    fn count_occurrences_works() {
        assert_eq!(count_occurrences("aaa", "a"), 3);
        assert_eq!(count_occurrences("abcabc", "abc"), 2);
        assert_eq!(count_occurrences("hello", "xyz"), 0);
        assert_eq!(count_occurrences("", "a"), 0);
        assert_eq!(count_occurrences("a", ""), 0);
    }
}
