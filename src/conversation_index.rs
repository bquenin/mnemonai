//! Persistent conversation metadata and search-text index.
//!
//! File-backed providers use this sidecar database to skip reparsing unchanged
//! JSONL transcripts and to reuse lowercased search text on warm starts.

use crate::history::{Conversation, PreviewPair, ProviderKind, path_to_string};
use chrono::{DateTime, Local};
use rusqlite::{Connection, TransactionBehavior, params};
use std::collections::HashMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// v3: dropped `show_last` from the primary key and split `preview` into
// `preview_first`/`preview_last`, so one row now serves both preview modes.
const SCHEMA_VERSION: i64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFingerprint {
    pub modified_millis: i64,
    pub size: i64,
}

/// A freshly parsed conversation awaiting persistence, borrowed for the save so
/// no `Conversation` is cloned. Both previews are stored regardless of the
/// caller's mode so a later `--first`/`--last` flip stays warm.
pub struct FreshRow<'a> {
    pub conversation: &'a Conversation,
    pub previews: &'a PreviewPair,
    pub fingerprint: SourceFingerprint,
}

/// One conversation resolved by a file provider during a scan: either reused
/// from the cache, or freshly parsed and (when a fingerprint is available)
/// awaiting persistence.
///
/// `Cached` also covers freshly parsed files that cannot be fingerprinted (a
/// failed `stat`): they are returned but not persisted, exactly as before.
pub enum LoadedConversation {
    Cached(Conversation),
    Fresh {
        conversation: Conversation,
        previews: PreviewPair,
        fingerprint: SourceFingerprint,
    },
}

impl LoadedConversation {
    /// Borrow the persistable payload, or `None` for entries that must not be
    /// written back (cache hits and unfingerprintable fresh parses).
    fn as_fresh(&self) -> Option<FreshRow<'_>> {
        match self {
            LoadedConversation::Fresh {
                conversation,
                previews,
                fingerprint,
            } => Some(FreshRow {
                conversation,
                previews,
                fingerprint: *fingerprint,
            }),
            LoadedConversation::Cached(_) => None,
        }
    }

    /// Consume into the conversation to return to the caller. Under the metadata
    /// profile (`include_full_text = false`) the full text is dropped *after*
    /// the save has read it, so the returned value stays lean while the cache
    /// row remains complete.
    pub fn into_conversation(self, include_full_text: bool) -> Conversation {
        let mut conversation = match self {
            LoadedConversation::Cached(conversation) => conversation,
            LoadedConversation::Fresh { conversation, .. } => conversation,
        };
        if !include_full_text {
            conversation.full_text = String::new();
        }
        conversation
    }
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
    include_full_text: bool,
) -> HashMap<PathBuf, CachedFileConversation> {
    let Some(conn) = open_index_db() else {
        return HashMap::new();
    };

    load_provider_cache_from_conn(&conn, provider, show_last, include_full_text).unwrap_or_default()
}

pub fn save_conversations<'a, I>(provider: ProviderKind, entries: I)
where
    I: IntoIterator<Item = FreshRow<'a>>,
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

    if let Err(err) = save_conversations_to_conn(&mut conn, provider, entries) {
        let _ = crate::debug_log::log_debug(&format!("conversation index: save failed: {err}"));
    }
}

fn save_conversations_to_conn<'a, I>(
    conn: &mut Connection,
    provider: ProviderKind,
    entries: I,
) -> rusqlite::Result<()>
where
    I: IntoIterator<Item = FreshRow<'a>>,
{
    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO file_conversations (
                schema_version, provider, source_path, source_mtime_millis,
                source_size, id, timestamp, preview_first, preview_last, full_text,
                project_name, project_path, cwd, message_count, parse_errors_json,
                summary, model, total_tokens, duration_minutes
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                ?15, ?16, ?17, ?18, ?19
            )",
        )?;

        let provider_key = provider.key();

        for FreshRow {
            conversation,
            previews,
            fingerprint,
        } in entries
        {
            let parse_errors_json = serde_json::to_string(&conversation.parse_errors)
                .unwrap_or_else(|_| "[]".to_string());

            stmt.execute(params![
                SCHEMA_VERSION,
                provider_key,
                conversation.path.to_string_lossy(),
                fingerprint.modified_millis,
                fingerprint.size,
                conversation.id,
                conversation.timestamp.to_rfc3339(),
                previews.first,
                previews.last,
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

/// Shared cache choreography for the file-backed providers (Claude, Codex,
/// Cursor Agent). Each provider keeps its own file enumeration and batching;
/// this owns the parts that were triplicated: loading the cache map honoring the
/// projection, consuming an entry on every load attempt, claiming files that
/// could not be fingerprinted so they are never mistaken for deletions, saving
/// fresh rows from references, and pruning whatever is left over once — and only
/// once — enumeration has succeeded.
pub struct ProviderCache {
    provider: ProviderKind,
    entries: Mutex<HashMap<PathBuf, CachedFileConversation>>,
}

impl ProviderCache {
    /// Load the provider's cache map, selecting previews for `show_last` and
    /// decoding `full_text` only when `include_full_text`.
    pub fn load(provider: ProviderKind, show_last: bool, include_full_text: bool) -> Self {
        Self {
            provider,
            entries: Mutex::new(load_provider_cache(provider, show_last, include_full_text)),
        }
    }

    /// An empty cache (no rows), for the defensive path where a background
    /// cache-load thread cannot be joined: every file then looks like a miss and
    /// is reparsed, and — since nothing was claimed — nothing is pruned.
    pub fn empty(provider: ProviderKind) -> Self {
        Self {
            provider,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Consume the cache entry for `path` — on *any* attempt, hit or miss —
    /// returning the cached conversation only when the fingerprint still matches.
    /// Removing on every attempt is what lets [`Self::prune_unclaimed`] treat the
    /// leftover map as exactly the set of deleted files.
    pub fn take_if_fresh(
        &self,
        path: &Path,
        fingerprint: SourceFingerprint,
    ) -> Option<Conversation> {
        let entry = self
            .entries
            .lock()
            .ok()
            .and_then(|mut entries| entries.remove(path))?;
        entry.into_conversation_if_fresh(fingerprint)
    }

    /// Claim `path` without using it, so a later prune leaves its cached row
    /// alone. For files that could not be `stat`ed (no fingerprint) or failed to
    /// parse: the file is still on disk, so its cache row must not be pruned.
    pub fn claim(&self, path: &Path) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.remove(path);
        }
    }

    /// Persist freshly parsed rows from a borrowed iterator — no `Conversation`
    /// clones. Safe to call repeatedly (per project or per batch); an empty
    /// iterator does not even open the database.
    pub fn save_fresh<'a, I>(&self, loaded: I)
    where
        I: IntoIterator<Item = &'a LoadedConversation>,
    {
        save_conversations(
            self.provider,
            loaded.into_iter().filter_map(LoadedConversation::as_fresh),
        );
    }

    /// Prune cache rows for files nobody claimed — i.e. deleted files. Call this
    /// only after a successful enumeration; a failed enumeration must skip it
    /// entirely (the caller returns early) or the cache would be emptied. Paths
    /// beneath any `keep_under` directory are spared, for providers (Claude)
    /// whose individual projects can fail to load while the rest succeed.
    pub fn prune_unclaimed(self, keep_under: &[PathBuf]) {
        let provider = self.provider;
        let leftover = self.entries.into_inner().unwrap_or_default();
        let stale: Vec<PathBuf> = leftover
            .into_keys()
            .filter(|path| !keep_under.iter().any(|dir| path.starts_with(dir)))
            .collect();
        prune_conversations(provider, &stale);
    }
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
    include_full_text: bool,
) -> rusqlite::Result<HashMap<PathBuf, CachedFileConversation>> {
    let mut stmt = conn.prepare(
        "SELECT
            source_path, source_mtime_millis, source_size, id, timestamp,
            preview_first, preview_last, full_text, project_name, project_path,
            cwd, message_count, parse_errors_json, summary, model, total_tokens,
            duration_minutes
         FROM file_conversations
         WHERE schema_version = ?1 AND provider = ?2",
    )?;

    let rows = stmt.query_map(params![SCHEMA_VERSION, provider.key()], |row| {
        let source_path: String = row.get(0)?;
        let timestamp: String = row.get(4)?;
        let timestamp = DateTime::parse_from_rfc3339(&timestamp)
            .map(|ts| ts.with_timezone(&Local))
            .unwrap_or_else(|_| Local::now());
        // One row holds both previews; pick the one this scan asked for.
        let preview: String = if show_last { row.get(6)? } else { row.get(5)? };
        // Under the metadata profile the (potentially large) full_text column is
        // never touched — SQLite skips reading it and nothing is retained.
        let full_text = if include_full_text {
            row.get(7)?
        } else {
            String::new()
        };
        let message_count: i64 = row.get(11)?;
        let parse_errors_json: String = row.get(12)?;
        let parse_errors = serde_json::from_str(&parse_errors_json).unwrap_or_default();
        let total_tokens: i64 = row.get(15)?;
        let duration_minutes: Option<i64> = row.get(16)?;

        Ok((
            PathBuf::from(&source_path),
            CachedFileConversation {
                fingerprint: SourceFingerprint {
                    modified_millis: row.get(1)?,
                    size: row.get(2)?,
                },
                conversation: Conversation {
                    path: PathBuf::from(source_path),
                    provider,
                    id: row.get(3)?,
                    timestamp,
                    preview,
                    full_text,
                    project_name: row.get(8)?,
                    project_path: optional_path(row.get(9)?),
                    cwd: optional_path(row.get(10)?),
                    message_count: message_count.max(0) as usize,
                    parse_errors,
                    summary: row.get(13)?,
                    model: row.get(14)?,
                    total_tokens: total_tokens.max(0) as u64,
                    duration_minutes: duration_minutes.map(|v| v.max(0) as u64),
                },
            },
        ))
    })?;

    Ok(rows.filter_map(|row| row.ok()).collect())
}

fn open_index_db() -> Option<Connection> {
    let home = home::home_dir()?;
    let dir = home.join(".local").join("state").join("mnemonai");
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
    "source_path",
    "source_mtime_millis",
    "source_size",
    "id",
    "timestamp",
    "preview_first",
    "preview_last",
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
             source_path           TEXT NOT NULL,
             source_mtime_millis   INTEGER NOT NULL,
             source_size           INTEGER NOT NULL,
             id                    TEXT NOT NULL,
             timestamp             TEXT NOT NULL,
             preview_first         TEXT NOT NULL,
             preview_last          TEXT NOT NULL,
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
             PRIMARY KEY (schema_version, provider, source_path)
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
        }
    }

    fn make_previews() -> PreviewPair {
        PreviewPair {
            first: "First Preview".to_string(),
            last: "Last Preview".to_string(),
        }
    }

    /// Save a single conversation with a throwaway preview pair, for tests that
    /// only care about round-tripping non-preview fields.
    fn save_one(
        conn: &mut Connection,
        provider: ProviderKind,
        conversation: &Conversation,
        fingerprint: SourceFingerprint,
    ) {
        let previews = make_previews();
        save_conversations_to_conn(
            conn,
            provider,
            [FreshRow {
                conversation,
                previews: &previews,
                fingerprint,
            }],
        )
        .unwrap();
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

        save_one(&mut conn, ProviderKind::Claude, &conversation, fingerprint);

        let mut cache =
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true).unwrap();
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
    }

    #[test]
    fn saved_row_serves_both_preview_modes_and_projects_full_text() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&mut conn).unwrap();
        let conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        let previews = PreviewPair {
            first: "First Preview".to_string(),
            last: "Last Preview".to_string(),
        };
        let fingerprint = SourceFingerprint {
            modified_millis: 10,
            size: 20,
        };
        // A single write (no show_last) must satisfy loads in either mode.
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            [FreshRow {
                conversation: &conversation,
                previews: &previews,
                fingerprint,
            }],
        )
        .unwrap();

        // --first picks preview_first, --last picks preview_last, both fresh.
        let first = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true)
            .unwrap()
            .remove(&conversation.path)
            .unwrap()
            .into_conversation_if_fresh(fingerprint)
            .unwrap();
        assert_eq!(first.preview, "First Preview");
        assert_eq!(first.full_text, "Hello Cache Body");

        let last = load_provider_cache_from_conn(&conn, ProviderKind::Claude, true, true)
            .unwrap()
            .remove(&conversation.path)
            .unwrap()
            .into_conversation_if_fresh(fingerprint)
            .unwrap();
        assert_eq!(last.preview, "Last Preview");

        // The metadata profile returns an empty full_text but every other field
        // is identical to the full-profile load.
        let projected = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, false)
            .unwrap()
            .remove(&conversation.path)
            .unwrap()
            .into_conversation_if_fresh(fingerprint)
            .unwrap();
        assert_eq!(projected.full_text, "");
        assert_eq!(projected.preview, first.preview);
        assert_eq!(projected.id, first.id);
        assert_eq!(projected.message_count, first.message_count);
    }

    #[test]
    fn init_schema_rebuilds_drifted_table() {
        // Simulate a table created by an older build: same name, missing the
        // full_text column the current code requires.
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
        save_one(&mut conn, ProviderKind::Claude, &conversation, fingerprint);

        let cache =
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true).unwrap();
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
        save_one(&mut conn, ProviderKind::Claude, &conversation, fingerprint);

        // Re-running init_schema (a second app start) must not wipe the cache.
        init_schema(&mut conn).unwrap();

        let cache =
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true).unwrap();
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
        let previews = make_previews();
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            [
                FreshRow {
                    conversation: &stale,
                    previews: &previews,
                    fingerprint,
                },
                FreshRow {
                    conversation: &current,
                    previews: &previews,
                    fingerprint,
                },
            ],
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
        let cache =
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true).unwrap();
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

        // One row now serves both preview modes; deleting it must clear both.
        save_one(&mut conn, ProviderKind::Claude, &conversation, fingerprint);

        delete_conversation_from_conn(&conn, ProviderKind::Claude, &conversation.path).unwrap();

        assert!(
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true)
                .unwrap()
                .is_empty()
        );
        assert!(
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, true, true)
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
        let previews = make_previews();
        save_conversations_to_conn(
            &mut conn,
            ProviderKind::Claude,
            [
                FreshRow {
                    conversation: &pruned,
                    previews: &previews,
                    fingerprint,
                },
                FreshRow {
                    conversation: &kept,
                    previews: &previews,
                    fingerprint,
                },
            ],
        )
        .unwrap();
        // The same source path cached for another provider must survive a
        // Claude prune.
        save_one(&mut conn, ProviderKind::Codex, &pruned, fingerprint);

        prune_conversations_from_conn(
            &mut conn,
            ProviderKind::Claude,
            std::slice::from_ref(&pruned.path),
        )
        .unwrap();

        let claude =
            load_provider_cache_from_conn(&conn, ProviderKind::Claude, false, true).unwrap();
        assert!(!claude.contains_key(&pruned.path));
        assert!(claude.contains_key(&kept.path));
        let codex = load_provider_cache_from_conn(&conn, ProviderKind::Codex, false, true).unwrap();
        assert!(codex.contains_key(&pruned.path));
    }
}
