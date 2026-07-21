//! Persistent conversation metadata and search-text index.
//!
//! File-backed providers use this sidecar database to skip reparsing unchanged
//! JSONL transcripts and to reuse lowercased search text on warm starts.

use crate::history::{Conversation, ProviderKind, path_to_string};
use chrono::{DateTime, Local};
use rusqlite::{Connection, TransactionBehavior, params};
use std::collections::HashMap;
use std::fs::Metadata;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: i64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFingerprint {
    pub modified_millis: i64,
    pub size: i64,
}

#[derive(Clone)]
pub struct CachedFileConversation {
    fingerprint: SourceFingerprint,
    conversation: Conversation,
}

impl CachedFileConversation {
    /// Consume the cache entry, returning the conversation when the source file
    /// is unchanged. Moving (rather than cloning) matters: cached entries carry
    /// the conversation's full text.
    pub fn into_conversation_if_fresh(
        self,
        fingerprint: SourceFingerprint,
    ) -> Option<Conversation> {
        let fresh = self.fingerprint == fingerprint;
        fresh.then_some(self.conversation)
    }
}

pub fn fingerprint_from_metadata(metadata: &Metadata) -> SourceFingerprint {
    SourceFingerprint {
        modified_millis: system_time_to_millis(metadata.modified().unwrap_or(UNIX_EPOCH)),
        size: metadata.len().min(i64::MAX as u64) as i64,
    }
}

pub fn load_provider_cache(
    provider: ProviderKind,
    show_last: bool,
) -> HashMap<PathBuf, CachedFileConversation> {
    let Some(conn) = open_index_db() else {
        return HashMap::new();
    };

    load_provider_cache_from_conn(&conn, provider, show_last).unwrap_or_default()
}

pub fn save_conversations<'a, I>(provider: ProviderKind, show_last: bool, entries: I)
where
    I: IntoIterator<Item = (&'a Conversation, SourceFingerprint)>,
{
    let entries: Vec<_> = entries.into_iter().collect();
    if entries.is_empty() {
        // The common warm-start case: don't even open the database.
        return;
    }

    let Some(mut conn) = open_index_db() else {
        let _ = crate::debug_log::log_debug("conversation index: failed to open database for save");
        return;
    };

    if let Err(err) = save_conversations_to_conn(&mut conn, provider, show_last, entries) {
        let _ = crate::debug_log::log_debug(&format!("conversation index: save failed: {err}"));
    }
}

fn save_conversations_to_conn<'a, I>(
    conn: &mut Connection,
    provider: ProviderKind,
    show_last: bool,
    entries: I,
) -> rusqlite::Result<()>
where
    I: IntoIterator<Item = (&'a Conversation, SourceFingerprint)>,
{
    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO file_conversations (
                schema_version, provider, show_last, source_path, source_mtime_millis,
                source_size, id, timestamp, preview, full_text, project_name,
                project_path, cwd, message_count, parse_errors_json, summary, model,
                total_tokens, duration_minutes
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                ?15, ?16, ?17, ?18, ?19
            )",
        )?;

        let provider_key = provider.key();
        let show_last = i64::from(show_last);

        for (conversation, fingerprint) in entries {
            let parse_errors_json = serde_json::to_string(&conversation.parse_errors)
                .unwrap_or_else(|_| "[]".to_string());

            stmt.execute(params![
                SCHEMA_VERSION,
                provider_key,
                show_last,
                conversation.path.to_string_lossy(),
                fingerprint.modified_millis,
                fingerprint.size,
                conversation.id,
                conversation.timestamp.to_rfc3339(),
                conversation.preview,
                conversation.full_text,
                conversation.project_name.as_deref(),
                path_to_string(conversation.project_path.as_deref()),
                path_to_string(conversation.cwd.as_deref()),
                conversation.message_count as i64,
                parse_errors_json,
                conversation.summary.as_deref(),
                conversation.model.as_deref(),
                conversation.total_tokens.min(i64::MAX as u64) as i64,
                conversation
                    .duration_minutes
                    .map(|v| v.min(i64::MAX as u64) as i64),
            ])?;
        }
    }

    tx.commit()
}

pub fn delete_conversation(provider: ProviderKind, path: &std::path::Path) {
    let Some(conn) = open_index_db() else {
        return;
    };

    let _ = delete_conversation_from_conn(&conn, provider, path);
}

/// Remove cached rows for source files that no longer exist. Callers pass the
/// paths left over in the cache map after a full provider scan consumed every
/// current file — anything still in the map has disappeared from disk (or been
/// excluded), and keeping it would grow the database and slow every load.
/// Only call this after a complete scan, never for a single-project load.
pub fn prune_conversations(provider: ProviderKind, paths: &[PathBuf]) {
    if paths.is_empty() {
        return;
    }

    let Some(mut conn) = open_index_db() else {
        return;
    };

    let _ = prune_conversations_from_conn(&mut conn, provider, paths);
}

fn prune_conversations_from_conn(
    conn: &mut Connection,
    provider: ProviderKind,
    paths: &[PathBuf],
) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;

    {
        let mut stmt =
            tx.prepare("DELETE FROM file_conversations WHERE provider = ?1 AND source_path = ?2")?;
        let provider_key = provider.key();

        for path in paths {
            stmt.execute(params![provider_key, path.to_string_lossy()])?;
        }
    }

    tx.commit()
}

fn delete_conversation_from_conn(
    conn: &Connection,
    provider: ProviderKind,
    path: &std::path::Path,
) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM file_conversations WHERE provider = ?1 AND source_path = ?2",
        params![provider.key(), path.to_string_lossy()],
    )?;
    Ok(())
}

fn load_provider_cache_from_conn(
    conn: &Connection,
    provider: ProviderKind,
    show_last: bool,
) -> rusqlite::Result<HashMap<PathBuf, CachedFileConversation>> {
    let mut stmt = conn.prepare(
        "SELECT
            source_path, source_mtime_millis, source_size, id, timestamp, preview,
            full_text, project_name, project_path, cwd, message_count,
            parse_errors_json, summary, model, total_tokens, duration_minutes
         FROM file_conversations
         WHERE schema_version = ?1 AND provider = ?2 AND show_last = ?3",
    )?;

    let rows = stmt.query_map(
        params![SCHEMA_VERSION, provider.key(), i64::from(show_last)],
        |row| {
            let source_path: String = row.get(0)?;
            let timestamp: String = row.get(4)?;
            let timestamp = DateTime::parse_from_rfc3339(&timestamp)
                .map(|ts| ts.with_timezone(&Local))
                .unwrap_or_else(|_| Local::now());
            let message_count: i64 = row.get(10)?;
            let parse_errors_json: String = row.get(11)?;
            let parse_errors = serde_json::from_str(&parse_errors_json).unwrap_or_default();
            let total_tokens: i64 = row.get(14)?;
            let duration_minutes: Option<i64> = row.get(15)?;

            Ok((
                PathBuf::from(&source_path),
                CachedFileConversation {
                    fingerprint: SourceFingerprint {
                        modified_millis: row.get(1)?,
                        size: row.get(2)?,
                    },
                    conversation: Conversation {
                        path: PathBuf::from(source_path),
                        index: 0,
                        provider: provider.clone(),
                        id: row.get(3)?,
                        timestamp,
                        preview: row.get(5)?,
                        full_text: row.get(6)?,
                        // Derived in parallel by precompute_search_text at
                        // startup; cheaper than reading them from disk.
                        search_text_lower: None,
                        search_topic_end: None,
                        project_name: row.get(7)?,
                        project_path: optional_path(row.get(8)?),
                        cwd: optional_path(row.get(9)?),
                        message_count: message_count.max(0) as usize,
                        parse_errors,
                        summary: row.get(12)?,
                        model: row.get(13)?,
                        total_tokens: total_tokens.max(0) as u64,
                        duration_minutes: duration_minutes.map(|v| v.max(0) as u64),
                    },
                },
            ))
        },
    )?;

    Ok(rows.filter_map(|row| row.ok()).collect())
}

fn open_index_db() -> Option<Connection> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(home)
        .join(".local")
        .join("state")
        .join("mnemonai");
    std::fs::create_dir_all(&dir).ok()?;
    let mut conn = Connection::open(dir.join("conversation_index.db")).ok()?;
    // Providers load and save in parallel threads; wait for the write lock
    // instead of failing with SQLITE_BUSY and silently dropping the batch.
    conn.busy_timeout(std::time::Duration::from_secs(5)).ok()?;
    init_schema(&mut conn).ok()?;
    Some(conn)
}

/// Every column the current code reads and writes. Used to detect schema drift.
const EXPECTED_COLUMNS: [&str; 19] = [
    "schema_version",
    "provider",
    "show_last",
    "source_path",
    "source_mtime_millis",
    "source_size",
    "id",
    "timestamp",
    "preview",
    "full_text",
    "project_name",
    "project_path",
    "cwd",
    "message_count",
    "parse_errors_json",
    "summary",
    "model",
    "total_tokens",
    "duration_minutes",
];

const CREATE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS file_conversations (
             schema_version        INTEGER NOT NULL,
             provider              TEXT NOT NULL,
             show_last             INTEGER NOT NULL,
             source_path           TEXT NOT NULL,
             source_mtime_millis   INTEGER NOT NULL,
             source_size           INTEGER NOT NULL,
             id                    TEXT NOT NULL,
             timestamp             TEXT NOT NULL,
             preview               TEXT NOT NULL,
             full_text             TEXT NOT NULL,
             project_name          TEXT,
             project_path          TEXT,
             cwd                   TEXT,
             message_count         INTEGER NOT NULL,
             parse_errors_json     TEXT NOT NULL,
             summary               TEXT,
             model                 TEXT,
             total_tokens          INTEGER NOT NULL,
             duration_minutes      INTEGER,
             PRIMARY KEY (schema_version, provider, show_last, source_path)
         );";

fn init_schema(conn: &mut Connection) -> rusqlite::Result<()> {
    // synchronous=NORMAL is safe with WAL (no corruption on crash) and makes
    // the large cold-start commit considerably cheaper than FULL.
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
    // CREATE TABLE IF NOT EXISTS silently keeps whatever shape an older build
    // created, after which every INSERT and SELECT here fails with "no such
    // column" — and the cache never reads or writes again. Validate the column
    // set and rebuild on drift; this is a cache, so the only cost is one
    // re-parse of changed providers.
    if !table_matches_expected_columns(conn)? {
        // Re-check under the write lock: providers open this database
        // concurrently at startup, and the check-then-rebuild must be atomic
        // so a stale check can never drop a table another connection just
        // rebuilt (and possibly populated). Exactly one connection rebuilds;
        // the rest see the fresh table here and leave it alone.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let drifted = !table_matches_expected_columns(&tx)?;
        if drifted {
            tx.execute_batch("DROP TABLE IF EXISTS file_conversations;")?;
            tx.execute_batch(CREATE_TABLE_SQL)?;
        }
        tx.commit()?;

        if drifted {
            // Return the dropped table's pages to the filesystem; without
            // this a previously bloated database file keeps its size forever.
            // Best-effort: VACUUM cannot run inside the transaction above.
            let _ = conn.execute_batch("VACUUM;");
        }
    }

    // Prune and single-file deletes filter on provider + source_path, which
    // is not a prefix of the primary key, so without this index each of those
    // DELETEs scans the whole table. Dropping the table above also drops the
    // index; recreate it after the drift check.
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_file_conversations_provider_path
             ON file_conversations (provider, source_path);",
    )?;

    // Rows are keyed by schema_version, so a version bump that keeps the
    // column set unchanged (no drift rebuild) strands the previous version's
    // rows, full_text payloads included: loads filter them out, saves write
    // new rows beside them, and prune never matches them. Purge them here; on
    // a fresh or just-rebuilt table this deletes nothing.
    conn.execute(
        "DELETE FROM file_conversations WHERE schema_version != ?1",
        params![SCHEMA_VERSION],
    )?;
    Ok(())
}

fn table_matches_expected_columns(conn: &Connection) -> rusqlite::Result<bool> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info('file_conversations')")?;
    let mut existing = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    existing.sort_unstable();

    let mut expected = EXPECTED_COLUMNS;
    expected.sort_unstable();

    Ok(existing.len() == expected.len()
        && existing
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied()))
}

fn system_time_to_millis(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

fn optional_path(path: Option<String>) -> Option<PathBuf> {
    path.filter(|path| !path.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Local;

    fn make_conversation(path: PathBuf) -> Conversation {
        Conversation {
            path,
            index: 0,
            provider: ProviderKind::Claude,
            id: "session-1".to_string(),
            timestamp: Local::now(),
            preview: "Hello Cache".to_string(),
            full_text: "Hello Cache Body".to_string(),
            project_name: Some("project".to_string()),
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: Vec::new(),
            summary: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
            search_text_lower: None,
            search_topic_end: None,
        }
    }

    #[test]
    fn load_cache_round_trips_conversation() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let mut conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        conversation.parse_errors.push(crate::history::ParseError {
            line_number: 7,
            line_content: "{bad json".to_string(),
            error_message: "expected value".to_string(),
            context_before: vec!["before".to_string()],
            context_after: vec!["after".to_string()],
        });
        let fingerprint = SourceFingerprint {
            modified_millis: 123,
            size: 456,
        };

        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        let mut cache = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        let cached = cache.remove(&conversation.path).unwrap();

        assert!(
            cached
                .clone()
                .into_conversation_if_fresh(SourceFingerprint {
                    modified_millis: 999,
                    size: 456
                })
                .is_none()
        );

        let loaded = cached.into_conversation_if_fresh(fingerprint).unwrap();
        assert_eq!(loaded.id, "session-1");
        assert_eq!(loaded.parse_errors.len(), 1);
        assert_eq!(loaded.parse_errors[0].line_number, 7);
        assert_eq!(loaded.full_text, "Hello Cache Body");
        // Search text is recomputed at startup rather than persisted.
        assert!(loaded.search_text_lower.is_none());
    }

    #[test]
    fn init_schema_rebuilds_drifted_table() {
        // Simulate a table created by an older build: same name, missing the
        // full_text/search_text_lower columns the current code requires.
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE file_conversations (
                schema_version INTEGER NOT NULL,
                provider       TEXT NOT NULL,
                source_path    TEXT NOT NULL,
                preview        TEXT NOT NULL,
                PRIMARY KEY (schema_version, provider, source_path)
            );",
        )
        .unwrap();

        init_schema(&mut conn).unwrap();

        // Save + load must round-trip after the rebuild.
        let conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        let fingerprint = SourceFingerprint {
            modified_millis: 1,
            size: 2,
        };
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        let cache = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        assert!(cache.contains_key(&conversation.path));
    }

    #[test]
    fn init_schema_preserves_rows_when_schema_matches() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        let fingerprint = SourceFingerprint {
            modified_millis: 1,
            size: 2,
        };
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        // Re-running init_schema (a second app start) must not wipe the cache.
        init_schema(&mut conn).unwrap();

        let cache = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        assert!(cache.contains_key(&conversation.path));
    }

    #[test]
    fn init_schema_purges_rows_from_other_schema_versions() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let stale = make_conversation(PathBuf::from("/tmp/stale.jsonl"));
        let current = make_conversation(PathBuf::from("/tmp/current.jsonl"));
        let fingerprint = SourceFingerprint {
            modified_millis: 1,
            size: 2,
        };
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&stale, fingerprint), (&current, fingerprint)],
        )
        .unwrap();
        // Rewrite one row as if a previous build with the same column set (so
        // no drift rebuild) had written it under an older SCHEMA_VERSION.
        conn.execute(
            "UPDATE file_conversations SET schema_version = ?1 WHERE source_path = ?2",
            params![SCHEMA_VERSION - 1, stale.path.to_string_lossy()],
        )
        .unwrap();

        // The next open must purge the old-version row and keep the current
        // one; without the purge it would linger in the file forever.
        init_schema(&mut conn).unwrap();

        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM file_conversations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(remaining, 1);
        let cache = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        assert!(cache.contains_key(&current.path));
    }

    #[test]
    fn delete_conversation_removes_cached_rows_for_all_preview_modes() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        let fingerprint = SourceFingerprint {
            modified_millis: 123,
            size: 456,
        };

        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            true,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        delete_conversation_from_conn(&conn, ProviderKind::Claude, &conversation.path).unwrap();

        assert!(
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false)
                .unwrap()
                .is_empty()
        );
        assert!(
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, true)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn prune_removes_only_the_given_paths_for_the_given_provider() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let pruned = make_conversation(PathBuf::from("/tmp/pruned.jsonl"));
        let kept = make_conversation(PathBuf::from("/tmp/kept.jsonl"));
        let fingerprint = SourceFingerprint {
            modified_millis: 1,
            size: 2,
        };
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            false,
            [(&pruned, fingerprint), (&kept, fingerprint)],
        )
        .unwrap();
        // The same source path cached for another provider must survive a
        // Claude prune.
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Codex,
            false,
            [(&pruned, fingerprint)],
        )
        .unwrap();

        prune_conversations_from_conn(
            &mut conn,
            ProviderKind::Claude,
            std::slice::from_ref(&pruned.path),
        )
        .unwrap();

        let claude = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        assert!(!claude.contains_key(&pruned.path));
        assert!(claude.contains_key(&kept.path));
        let codex = load_provider_cache_from_conn(&conn, ProviderKind::Codex, false).unwrap();
        assert!(codex.contains_key(&pruned.path));
    }
}
