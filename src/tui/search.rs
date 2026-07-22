use crate::history::Conversation;
use chrono::{DateTime, Local};
use rayon::prelude::*;

/// Size of the "topic window" — the first ~2000 characters of a conversation,
/// covering the initial exchanges where the user establishes intent.
const TOPIC_WINDOW_SIZE: usize = 2000;

/// How much extra weight topic-window matches get over body matches.
const TOPIC_WEIGHT: f64 = 3.0;

/// Precomputed search data for a conversation
pub struct SearchableConversation {
    /// Lowercased full text for searching. This is the only in-memory copy of
    /// the conversation body: `Conversation.full_text` is emptied during
    /// precompute so the corpus is not held twice. The list UI derives its
    /// hidden-match context line from this lowercased text.
    pub text_lower: String,
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

/// Check if a character is a word boundary (non-alphanumeric).
fn is_word_boundary(c: char) -> bool {
    !c.is_alphanumeric() && c != '_'
}

/// A query term with metadata about whether it's "completed" (followed by whitespace).
/// Completed terms require a word boundary after the match in the text.
#[derive(Clone, Debug)]
struct QueryTerm<'a> {
    text: &'a str,
    /// True if this term was followed by whitespace in the query,
    /// meaning the user finished typing it and it should match as a whole word.
    completed: bool,
}

/// Find a char-boundary at or after TOPIC_WINDOW_SIZE bytes.
fn topic_end_for_text(text: &str) -> usize {
    if text.len() <= TOPIC_WINDOW_SIZE {
        text.len()
    } else {
        let mut end = TOPIC_WINDOW_SIZE;
        while !text.is_char_boundary(end) && end < text.len() {
            end += 1;
        }
        end
    }
}

/// Precompute lowercased search text for all conversations.
/// Takes `full_text` out of each Conversation, lowercases it into
/// `text_lower`, and drops the original — so the conversation body lives in
/// memory exactly once (as the lowercased search copy).
pub fn precompute_search_text(conversations: &mut [Conversation]) -> Vec<SearchableConversation> {
    conversations
        .par_iter_mut()
        .enumerate()
        .map(|(idx, conv)| {
            let full_text = std::mem::take(&mut conv.full_text);
            let text_lower = normalize_for_search(&full_text);
            let topic_end = topic_end_for_text(&text_lower);
            SearchableConversation {
                text_lower,
                topic_end,
                index: idx,
            }
        })
        .collect()
}

/// Filter and score conversations based on query.
/// Returns indices into the original conversations vec, sorted by score descending.
/// When `narrow_from` is provided, only scores those indices (used when query extends previous).
/// Parse query into terms, tracking whether the last term is "completed" (has trailing whitespace).
/// A trailing space means the user finished typing the last term and it should match as a whole word.
/// Interior spaces are just term separators — those terms remain prefix-matchable.
fn parse_query_terms(query_lower: &str) -> Vec<QueryTerm<'_>> {
    let has_trailing_space = query_lower.ends_with(|c: char| c.is_whitespace());
    let terms: Vec<&str> = query_lower.split_whitespace().collect();
    terms
        .iter()
        .enumerate()
        .map(|(i, &text)| QueryTerm {
            text,
            // Only the last term can be "completed" — trailing space signals the user
            // finished typing it. Interior spaces are just term separators.
            completed: i == terms.len() - 1 && has_trailing_space,
        })
        .collect()
}

pub fn search(
    conversations: &[Conversation],
    searchable: &[SearchableConversation],
    query: &str,
    now: DateTime<Local>,
    narrow_from: Option<&[usize]>,
) -> Vec<usize> {
    search_scored(conversations, searchable, query, now, narrow_from)
        .into_iter()
        .map(|(idx, _)| idx)
        .collect()
}

/// Like [`search`], but also returns each conversation's relevance score.
///
/// This is the single ranking implementation; [`search`] is a thin wrapper that
/// drops the scores. The headless `search` command uses the scores directly, and
/// a parity test locks the two in step so ranking is never forked.
///
/// Results are `(conversation index, score)` pairs ordered by score descending,
/// then timestamp descending. When the query is empty every conversation is
/// returned with a `0.0` score (mirroring [`search`]'s "return all" behavior).
pub fn search_scored(
    conversations: &[Conversation],
    searchable: &[SearchableConversation],
    query: &str,
    now: DateTime<Local>,
    narrow_from: Option<&[usize]>,
) -> Vec<(usize, f64)> {
    if query.trim().is_empty() {
        // Return all indices sorted by timestamp (already sorted in history.rs)
        return (0..conversations.len()).map(|idx| (idx, 0.0)).collect();
    }

    let query_lower = normalize_for_search(query);

    // Score conversations in parallel, optionally narrowing to a subset
    let query_terms = parse_query_terms(&query_lower);
    if query_terms.is_empty() {
        return (0..conversations.len()).map(|idx| (idx, 0.0)).collect();
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

    scored
        .into_iter()
        .map(|(idx, score, _)| (idx, score))
        .collect()
}

/// Locate every query-term occurrence in `text_lower`, as `(byte_offset,
/// char_len)` pairs sorted ascending by offset.
///
/// `text_lower` must already be lowercased (it is `SearchableConversation::
/// text_lower`); `query` is normalized here. Term parsing and the whole-word
/// rule match [`search`]/`score_text` exactly: interior terms and a final term
/// without a trailing space match as substrings, while a final term with a
/// trailing space matches only at a word boundary. Occurrences of a single term
/// are non-overlapping (advancing past each match), so the returned length for
/// one term equals `count_occurrences` over the whole text. The headless search
/// command uses this for `match_count` and to anchor snippet windows.
pub fn match_offsets(text_lower: &str, query: &str) -> Vec<(usize, usize)> {
    let query_lower = normalize_for_search(query);
    let query_terms = parse_query_terms(&query_lower);

    let mut offsets = Vec::new();
    for term in &query_terms {
        if term.text.is_empty() {
            continue;
        }
        let char_len = term.text.chars().count();
        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(term.text) {
            let abs = start + pos;
            let abs_end = abs + term.text.len();
            let at_boundary = !term.completed
                || abs_end >= text_lower.len()
                || text_lower[abs_end..]
                    .chars()
                    .next()
                    .is_some_and(is_word_boundary);
            if at_boundary {
                offsets.push((abs, char_len));
            }
            // Advance past the whole match (non-overlapping), matching
            // `count_occurrences` so `match_count` stays consistent.
            start = abs_end;
        }
    }
    offsets.sort_unstable_by_key(|&(pos, _)| pos);
    offsets
}

/// Count non-overlapping occurrences of `needle` in `haystack`.
/// If `whole_word` is true, only counts matches where the character after the match
/// is a word boundary (or end of string).
fn count_occurrences(haystack: &str, needle: &str, whole_word: bool) -> usize {
    if needle.is_empty() {
        return 0;
    }
    if !whole_word {
        return haystack.matches(needle).count();
    }
    let mut count = 0;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let abs_end = start + pos + needle.len();
        // Check that the character after the match is a word boundary (or end of string)
        let at_boundary = abs_end >= haystack.len()
            || haystack[abs_end..]
                .chars()
                .next()
                .is_some_and(is_word_boundary);
        if at_boundary {
            count += 1;
        }
        start = abs_end;
    }
    count
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
    query_terms: &[QueryTerm<'_>],
    timestamp: DateTime<Local>,
    now: DateTime<Local>,
) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let topic_window = &text_lower[..topic_end];
    let body = &text_lower[topic_end..];

    let mut relevance = 0.0;
    for term in query_terms {
        let topic_hits = count_occurrences(topic_window, term.text, term.completed);
        let body_hits = count_occurrences(body, term.text, term.completed);
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

/// Recency boost half-life: a conversation this old gets half the boost.
const RECENCY_HALF_LIFE_DAYS: f64 = 14.0;

/// Maximum recency boost, applied at age zero and decaying smoothly toward 0.
/// Capped at 1.0 (a 2x multiplier) so recency breaks ties between similarly
/// relevant conversations instead of overriding topical relevance: per-term
/// relevance is sqrt-dampened, so a larger boost lets today's passing mention
/// outrank an older conversation that is actually about the topic.
const RECENCY_MAX_BOOST: f64 = 1.0;

/// Calculate recency multiplier based on age.
///
/// Smooth exponential decay rather than day/week/month steps: step cliffs
/// reshuffled rankings whenever a conversation aged across a bucket boundary
/// (an 8-day-old conversation was penalized 25% against a 6-day-old one).
fn recency_multiplier(timestamp: DateTime<Local>, now: DateTime<Local>) -> f64 {
    // Clamp future timestamps (clock skew) to "now".
    let age_seconds = now.signed_duration_since(timestamp).num_seconds().max(0);
    let age_days = age_seconds as f64 / 86_400.0;
    1.0 + RECENCY_MAX_BOOST * 0.5_f64.powf(age_days / RECENCY_HALF_LIFE_DAYS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Conversation;
    use chrono::Duration;
    use std::path::PathBuf;

    fn make_conv(text: &str, timestamp: DateTime<Local>) -> Conversation {
        Conversation {
            path: PathBuf::new(),
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
    fn recency_boost_decays_monotonically() {
        let now = Local::now();
        let m = |days: i64| recency_multiplier(now - Duration::days(days), now);
        assert!(m(0) > m(1));
        assert!(m(1) > m(7));
        assert!(m(7) > m(30));
        assert!(m(30) > m(90));
        assert!(m(365) >= 1.0);
    }

    #[test]
    fn recency_boost_is_bounded() {
        let now = Local::now();
        assert_eq!(recency_multiplier(now, now), 1.0 + RECENCY_MAX_BOOST);
        let very_old = recency_multiplier(now - Duration::days(3650), now);
        assert!(very_old >= 1.0);
        assert!(very_old < 1.001);
    }

    #[test]
    fn recency_halves_the_boost_at_the_half_life() {
        let now = Local::now();
        // Derive the age in seconds so the test stays correct if the constant
        // is ever changed to a fractional number of days.
        let half_life = Duration::seconds((RECENCY_HALF_LIFE_DAYS * 86_400.0).round() as i64);
        let at_half_life = recency_multiplier(now - half_life, now);
        let expected = 1.0 + RECENCY_MAX_BOOST / 2.0;
        assert!((at_half_life - expected).abs() < 1e-6);
    }

    #[test]
    fn recency_has_no_cliffs() {
        let now = Local::now();
        // The old step function dropped the boost 33% between these two ages.
        let before = recency_multiplier(now - Duration::hours(23), now);
        let after = recency_multiplier(now - Duration::hours(25), now);
        assert!(
            (before - after) / before < 0.01,
            "aging two hours should barely change the boost: {before} vs {after}"
        );
    }

    /// The complaint that motivated smooth decay: a conversation that is
    /// actually about the topic but fell off the old 30-day cliff lost to a
    /// conversation from today that mentions the term once in passing.
    #[test]
    fn topical_month_old_convo_outranks_recent_passing_mention() {
        let now = Local::now();
        let mut convs = vec![
            make_conv(
                "grpc retries grpc deadlines grpc pooling grpc keepalive notes",
                now - Duration::days(35),
            ),
            make_conv(
                &format!(
                    "planning notes {} grpc might be worth a look",
                    "filler text ".repeat(70)
                ),
                now,
            ),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "grpc", now, None);
        assert_eq!(
            results[0], 0,
            "the conversation about grpc should outrank today's passing mention"
        );
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
    fn future_timestamp_clamps_to_max_boost() {
        let now = Local::now();
        let timestamp = now + Duration::hours(1);
        assert_eq!(recency_multiplier(timestamp, now), 1.0 + RECENCY_MAX_BOOST);
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
            make_conv("deploy deploy deploy the deploy fix for deploy", now),
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
            make_conv(&format!("{padding} someone mentioned webpack once"), now),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "webpack", now, None);
        assert_eq!(results.len(), 2);
        // The highly relevant old conversation should beat the barely-relevant recent one
        assert_eq!(
            results[0], 0,
            "highly relevant old convo should rank above barely relevant recent one"
        );
    }

    #[test]
    fn topic_window_match_ranks_higher_than_body_match() {
        let now = Local::now();
        // Build two conversations with the same total number of "migrate" hits,
        // but one has hits in the topic window (first ~2000 chars) and the other in the body.
        let padding = "x ".repeat(1500); // ~3000 chars of padding
        let mut convs = vec![
            // "migrate" only in the body (after the topic window)
            make_conv(&format!("{padding}migrate migrate migrate"), now),
            // "migrate" in the topic window (beginning of conversation)
            make_conv(&format!("migrate migrate migrate {padding}"), now),
        ];
        let searchable = precompute_search_text(&mut convs);
        let results = search(&convs, &searchable, "migrate", now, None);
        assert_eq!(results.len(), 2);
        // The conversation with topic-window hits should rank first
        assert_eq!(
            results[0], 1,
            "topic-window match should rank higher than body-only match"
        );
    }

    #[test]
    fn count_occurrences_works() {
        // Substring mode (whole_word = false)
        assert_eq!(count_occurrences("aaa", "a", false), 3);
        assert_eq!(count_occurrences("abcabc", "abc", false), 2);
        assert_eq!(count_occurrences("hello", "xyz", false), 0);
        assert_eq!(count_occurrences("", "a", false), 0);
        assert_eq!(count_occurrences("a", "", false), 0);
    }

    #[test]
    fn count_occurrences_whole_word() {
        // "dia" as whole word should not match "diagnostics"
        assert_eq!(count_occurrences("diagnostics are useful", "dia", true), 0);
        // "dia" as whole word should match "dia" followed by space
        assert_eq!(
            count_occurrences("dia is short for diagram", "dia", true),
            1
        );
        // "dia" at end of string
        assert_eq!(count_occurrences("this is dia", "dia", true), 1);
        // "dia" followed by punctuation
        assert_eq!(count_occurrences("check dia, then move on", "dia", true), 1);
        // Multiple whole-word matches
        assert_eq!(count_occurrences("dia and dia again", "dia", true), 2);
    }

    #[test]
    fn trailing_space_filters_prefix_matches() {
        let now = Local::now();
        let mut convs = vec![
            // Contains "diagnostics" but not "dia" as a standalone word
            make_conv("run diagnostics on the server", now),
            // Contains "dia" as a standalone word
            make_conv("the dia tool is useful for diagrams", now),
        ];
        let searchable = precompute_search_text(&mut convs);

        // Without trailing space: both match (substring "dia" in "diagnostics")
        let results = search(&convs, &searchable, "dia", now, None);
        assert_eq!(results.len(), 2, "prefix search should match both");

        // With trailing space: only the standalone "dia" matches
        let results = search(&convs, &searchable, "dia ", now, None);
        assert_eq!(
            results.len(),
            1,
            "completed term should only match whole word"
        );
        assert_eq!(
            results[0], 1,
            "should match the conversation with standalone 'dia'"
        );
    }

    #[test]
    fn parse_query_terms_tracks_completion() {
        // Only the last term is affected by trailing space; interior terms are always prefix-matchable
        let terms = parse_query_terms("foo bar");
        assert_eq!(terms.len(), 2);
        assert!(!terms[0].completed, "interior term is never completed");
        assert!(
            !terms[1].completed,
            "last term without trailing space should not be completed"
        );

        let terms = parse_query_terms("foo bar ");
        assert_eq!(terms.len(), 2);
        assert!(!terms[0].completed, "interior term is never completed");
        assert!(
            terms[1].completed,
            "last term with trailing space should be completed"
        );

        let terms = parse_query_terms("single");
        assert_eq!(terms.len(), 1);
        assert!(!terms[0].completed);

        let terms = parse_query_terms("single ");
        assert_eq!(terms.len(), 1);
        assert!(terms[0].completed);
    }

    #[test]
    fn search_and_search_scored_agree_on_order() {
        // The scored variant is the single ranking implementation; `search`
        // must return exactly its indices in the same order.
        let now = Local::now();
        let mut convs = vec![
            make_conv("deploy deploy deploy the deploy fix", now),
            make_conv(&format!("deploy once {}", "filler ".repeat(120)), now),
            make_conv("nothing relevant here", now),
        ];
        let searchable = precompute_search_text(&mut convs);

        let order = search(&convs, &searchable, "deploy", now, None);
        let scored = search_scored(&convs, &searchable, "deploy", now, None);
        let scored_order: Vec<usize> = scored.iter().map(|&(idx, _)| idx).collect();

        assert_eq!(order, scored_order);
        // Scores are strictly descending in the returned order.
        for pair in scored.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn match_offsets_counts_and_locates_terms() {
        // Two terms; occurrences of each are located and the total equals the
        // per-term whole-text counts. Offsets are sorted ascending.
        let text = normalize_for_search("workflow secrets and more workflow notes");
        let offsets = match_offsets(&text, "workflow secret");

        // "workflow" (prefix) appears twice, "secret" (prefix of "secrets") once.
        assert_eq!(offsets.len(), 3);
        assert!(offsets.windows(2).all(|w| w[0].0 <= w[1].0));
        // Every reported offset actually starts one of the query terms.
        for &(pos, char_len) in &offsets {
            let slice: String = text[pos..].chars().take(char_len).collect();
            assert!(slice == "workflow" || slice == "secret");
        }
    }

    #[test]
    fn match_offsets_respects_trailing_space_whole_word() {
        let text = normalize_for_search("run diagnostics; the dia tool helps");
        // Prefix term "dia" matches inside "diagnostics" and the standalone "dia".
        assert_eq!(match_offsets(&text, "dia").len(), 2);
        // Completed term "dia " (trailing space) only matches the whole word.
        assert_eq!(match_offsets(&text, "dia ").len(), 1);
    }
}
