use super::LoadOptions;
use crate::claude::{AssistantMessage, ContentBlock, LogEntry, UserContent, UserMessage};
use crate::error::{AppError, Result};
use crate::history::{Conversation, LoaderMessage, ProviderKind};
use chrono::{DateTime, Local, TimeZone, Utc};
use rayon::prelude::*;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};

const EXTENSION_ID: &str = "mnemonai.mnemonai-bridge";

/// Bubble type constants from Cursor's data model
const BUBBLE_TYPE_USER: i64 = 1;
const BUBBLE_TYPE_ASSISTANT: i64 = 2;

pub struct CursorProvider {
    /// Path to the global state database: ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb
    global_db_path: PathBuf,
    /// Path to workspace storage: ~/Library/Application Support/Cursor/User/workspaceStorage/
    workspace_storage_path: PathBuf,
}

#[derive(Deserialize)]
struct ConversationIndexEntry {
    #[serde(rename = "conversationId")]
    conversation_id: String,
    timestamp: i64,
}

/// Workspace information extracted from per-workspace composer data.
struct WorkspaceInfo {
    /// Directory path for display/filtering (parent dir for .code-workspace files)
    path: PathBuf,
    /// Path to pass to `cursor` CLI for opening (may be a .code-workspace file)
    open_path: PathBuf,
    title: Option<String>,
    /// Timestamp in millis from composerData (lastUpdatedAt or createdAt)
    timestamp_millis: Option<i64>,
}

/// Per-conversation metadata collected from GROUP BY queries.
struct ConvInfo {
    conv_id: String,
    bubble_count: usize,
    first_key: String,
    preview_key: String,
    /// Key of a user-type bubble (from a second query filtered on type=1).
    /// This is the key fix: MIN/MAX(key) over all bubbles often picks an assistant
    /// bubble because keys contain random UUIDs. This field targets user bubbles specifically.
    user_preview_key: Option<String>,
}

impl CursorProvider {
    pub fn new() -> Self {
        let home = home::home_dir().unwrap_or_default();
        let cursor_user = home
            .join("Library")
            .join("Application Support")
            .join("Cursor")
            .join("User");
        Self {
            global_db_path: cursor_user.join("globalStorage").join("state.vscdb"),
            workspace_storage_path: cursor_user.join("workspaceStorage"),
        }
    }

    /// Read workspace.json from a workspace directory to get the project paths.
    /// Returns (display_path, open_path) where display_path is the directory
    /// (for filtering/display) and open_path is what to pass to `cursor` CLI
    /// (may be a .code-workspace file).
    fn resolve_workspace_path(workspace_dir: &Path) -> Option<(PathBuf, PathBuf)> {
        let workspace_json = workspace_dir.join("workspace.json");
        let content = std::fs::read_to_string(&workspace_json).ok()?;
        let json: Value = serde_json::from_str(&content).ok()?;
        // workspace.json may have "folder" or "workspace" key, both are file:// URIs
        let uri = json
            .get("folder")
            .or_else(|| json.get("workspace"))
            .and_then(|v| v.as_str())?;
        let path_str = uri.strip_prefix("file://")?;
        let open_path = PathBuf::from(path_str);
        // For .code-workspace files, use parent dir for display/filtering
        let display_path = if open_path
            .extension()
            .is_some_and(|ext| ext == "code-workspace")
        {
            open_path.parent()?.to_path_buf()
        } else {
            open_path.clone()
        };
        Some((display_path, open_path))
    }

    /// Build a map from conversation ID → (workspace path, optional title).
    /// Scans per-workspace state.vscdb files for composer.composerData entries
    /// which contain allComposers[].composerId and name values.
    fn build_workspace_map(&self) -> HashMap<String, WorkspaceInfo> {
        let workspace_dirs: Vec<_> = std::fs::read_dir(&self.workspace_storage_path)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        // Process workspaces in parallel
        let maps: Vec<Vec<(String, WorkspaceInfo)>> = workspace_dirs
            .par_iter()
            .filter_map(|entry| {
                let dir = entry.path();
                let (display_path, open_path) = Self::resolve_workspace_path(&dir)?;
                let db_path = dir.join("state.vscdb");
                if !db_path.exists() {
                    return None;
                }
                let conn = Connection::open_with_flags(
                    &db_path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .ok()?;

                // Read composer.composerData to get composerIds for this workspace
                let composer_data: String = conn
                    .query_row(
                        "SELECT value FROM ItemTable WHERE key = 'composer.composerData'",
                        [],
                        |row| row.get(0),
                    )
                    .ok()?;

                let parsed: Value = serde_json::from_str(&composer_data).ok()?;
                let composers = parsed.get("allComposers")?.as_array()?;

                let pairs: Vec<(String, WorkspaceInfo)> = composers
                    .iter()
                    .filter_map(|c| {
                        let id = c.get("composerId")?.as_str()?.to_string();
                        let title = c
                            .get("name")
                            .and_then(|n| n.as_str())
                            .filter(|s| !s.is_empty())
                            .map(String::from);
                        let timestamp_millis = c
                            .get("lastUpdatedAt")
                            .or_else(|| c.get("createdAt"))
                            .and_then(|v| v.as_i64());
                        Some((
                            id,
                            WorkspaceInfo {
                                path: display_path.clone(),
                                open_path: open_path.clone(),
                                title,
                                timestamp_millis,
                            },
                        ))
                    })
                    .collect();

                Some(pairs)
            })
            .collect();

        let mut map = HashMap::new();
        for pairs in maps {
            for (conv_id, info) in pairs {
                map.entry(conv_id).or_insert(info);
            }
        }
        map
    }

    /// Load all conversations from the global Cursor database.
    fn load_from_global_db(
        &self,
        show_last: bool,
        include_full_text: bool,
    ) -> Result<Vec<Conversation>> {
        let conn = Connection::open_with_flags(
            &self.global_db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        let workspace_map = self.build_workspace_map();
        load_conversations_from_conn(
            &conn,
            show_last,
            include_full_text,
            &workspace_map,
            &self.global_db_path,
        )
    }

    /// Load all bubbles for a conversation from the global database.
    fn load_bubbles(conv_id: &str, global_db_path: &Path) -> Result<Vec<Bubble>> {
        let conn = Connection::open_with_flags(
            global_db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        let prefix = format!("bubbleId:{}:", conv_id);
        let prefix_end = format!("bubbleId:{};", conv_id); // ';' is after ':' in ASCII

        let mut stmt = conn.prepare(
            "SELECT CAST(value AS TEXT) FROM cursorDiskKV WHERE key >= ? AND key < ? ORDER BY ROWID",
        )?;

        let bubbles: Vec<Bubble> = stmt
            .query_map(rusqlite::params![&prefix, &prefix_end], |row| {
                row.get::<_, String>(0)
            })?
            .filter_map(|r| r.ok())
            .filter_map(|json_str| parse_bubble(&json_str))
            .collect();

        Ok(bubbles)
    }
}

/// Enumerate all conversations with counts and boundary keys.
/// Only accesses keys (no value I/O), so this is a fast index scan. The
/// grouping happens in Rust: the ordered scan already delivers all keys of a
/// conversation contiguously, whereas GROUP BY on a SUBSTR() expression makes
/// SQLite materialize and sort every key a second time.
fn query_conv_infos(conn: &Connection, show_last: bool) -> Result<Vec<ConvInfo>> {
    let mut stmt = conn.prepare(
        "SELECT key FROM cursorDiskKV \
         WHERE key >= 'bubbleId:' AND key < 'bubbleId;' \
         ORDER BY key",
    )?;

    let mut infos: Vec<ConvInfo> = Vec::new();
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let Ok(key) = row.get_ref(0)?.as_str() else {
            continue;
        };
        let Some(conv_id) = conv_id_from_bubble_key(key) else {
            continue;
        };

        match infos.last_mut() {
            Some(info) if info.conv_id == conv_id => {
                info.bubble_count += 1;
                // MIN(key) is the first key of the group; MAX(key) the last.
                if show_last {
                    info.preview_key.clear();
                    info.preview_key.push_str(key);
                }
            }
            _ => infos.push(ConvInfo {
                conv_id: conv_id.to_string(),
                bubble_count: 1,
                first_key: key.to_string(),
                preview_key: key.to_string(),
                user_preview_key: None,
            }),
        }
    }

    Ok(infos)
}

/// Extract `<conv_id>` from a `bubbleId:<conv_id>:<bubble_id>` key.
fn conv_id_from_bubble_key(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("bubbleId:")?;
    let colon = rest.find(':')?;
    Some(&rest[..colon]).filter(|conv_id| !conv_id.is_empty())
}

/// Query for the first/last user-type bubble per conversation using json_extract.
/// This is the slowest step (~500ms) because it reads and parses JSON for every bubble row.
fn query_user_bubble_keys(conn: &Connection, show_last: bool) -> HashMap<String, String> {
    let order_func = if show_last { "MAX" } else { "MIN" };
    let user_query = format!(
        "SELECT SUBSTR(key, 10, INSTR(SUBSTR(key, 10), ':') - 1) as conv_id, \
                {}(key) as user_key \
         FROM cursorDiskKV \
         WHERE key >= 'bubbleId:' AND key < 'bubbleId;' \
           AND json_extract(CAST(value AS TEXT), '$.type') = 1 \
         GROUP BY conv_id",
        order_func
    );
    let mut user_keys = HashMap::new();
    if let Ok(mut stmt) = conn.prepare(&user_query)
        && let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
    {
        for row in rows.flatten() {
            user_keys.insert(row.0, row.1);
        }
    }
    user_keys
}

/// Open (or create) the sidecar cache database for user bubble keys.
/// Returns None on any error — callers fall back to the full query.
fn open_cache_db() -> Option<Connection> {
    let home = home::home_dir()?;
    let dir = home.join(".local").join("state").join("mnemonai");
    std::fs::create_dir_all(&dir).ok()?;
    let db_path = dir.join("cursor_cache.db");
    let conn = Connection::open(&db_path).ok()?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS user_bubble_keys (
             conv_id      TEXT PRIMARY KEY,
             min_user_key TEXT,
             max_user_key TEXT,
             bubble_count INTEGER
         );
         CREATE TABLE IF NOT EXISTS conversation_full_text (
             conv_id      TEXT PRIMARY KEY,
             full_text    TEXT NOT NULL,
             bubble_count INTEGER NOT NULL
         );",
    )
    .ok()?;
    // Migrate pre-existing user_bubble_keys tables that lack the bubble_count
    // column (added for staleness detection). Rows keep bubble_count NULL, which
    // reads back as a mismatch below and triggers a one-time re-derive.
    let _ = conn.execute_batch("ALTER TABLE user_bubble_keys ADD COLUMN bubble_count INTEGER");
    Some(conn)
}

/// A cached user-bubble-key entry: (min_user_key, max_user_key, bubble_count).
/// `bubble_count` is `None` for rows written before the column existed; such
/// rows never match a live conversation's count and so are treated as stale.
type CachedUserKeys = (Option<String>, Option<String>, Option<usize>);

/// Bulk-load all cached user bubble key entries.
fn load_cached_user_keys(cache_conn: &Connection) -> HashMap<String, CachedUserKeys> {
    let mut map = HashMap::new();
    let mut stmt = match cache_conn
        .prepare("SELECT conv_id, min_user_key, max_user_key, bubble_count FROM user_bubble_keys")
    {
        Ok(s) => s,
        Err(_) => return map,
    };
    let rows = match stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<i64>>(3)?.map(|v| v.max(0) as usize),
        ))
    }) {
        Ok(r) => r,
        Err(_) => return map,
    };
    for row in rows.flatten() {
        map.insert(row.0, (row.1, row.2, row.3));
    }
    map
}

/// Query MIN and MAX user-type bubble keys for a single conversation.
/// Uses the primary key index with a prefix range — sub-millisecond.
fn query_user_key_for_conv(
    cursor_conn: &Connection,
    conv_id: &str,
) -> (Option<String>, Option<String>) {
    let prefix = format!("bubbleId:{}:", conv_id);
    let prefix_end = format!("bubbleId:{};", conv_id);
    let Ok(mut stmt) = cursor_conn.prepare_cached(
        "SELECT MIN(key), MAX(key) FROM cursorDiskKV \
         WHERE key >= ?1 AND key < ?2 \
           AND json_extract(CAST(value AS TEXT), '$.type') = 1",
    ) else {
        return (None, None);
    };
    let result = stmt.query_row(rusqlite::params![&prefix, &prefix_end], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
        ))
    });
    result.unwrap_or((None, None))
}

/// Resolve user bubble keys for many conversations, in parallel across
/// read-only connections when the database can be reopened by path.
/// Returns (conv_id, min_user_key, max_user_key) tuples; conversations whose
/// chunk failed are simply absent.
fn query_user_keys_for_convs(
    db_path: &Path,
    fallback_conn: &Connection,
    conv_ids: &[String],
) -> Vec<(String, Option<String>, Option<String>)> {
    if conv_ids.len() <= 4 || open_global_ro(db_path).is_none() {
        return conv_ids
            .iter()
            .map(|conv_id| {
                let (min_key, max_key) = query_user_key_for_conv(fallback_conn, conv_id);
                (conv_id.clone(), min_key, max_key)
            })
            .collect();
    }

    conv_ids
        .par_chunks(32)
        .filter_map(|chunk| {
            let conn = open_global_ro(db_path)?;
            Some(
                chunk
                    .iter()
                    .map(|conv_id| {
                        let (min_key, max_key) = query_user_key_for_conv(&conn, conv_id);
                        (conv_id.clone(), min_key, max_key)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect()
}

/// A user-bubble-key entry to persist: (conv_id, min_user_key, max_user_key, bubble_count).
type UserKeyEntry = (String, Option<String>, Option<String>, usize);

/// Batch-insert user bubble key entries into the cache in a single transaction.
fn save_user_keys_to_cache(cache_conn: &Connection, entries: &[UserKeyEntry]) {
    if entries.is_empty() {
        return;
    }
    let _ = cache_conn.execute_batch("BEGIN");
    {
        let mut stmt = match cache_conn.prepare_cached(
            "INSERT OR REPLACE INTO user_bubble_keys (conv_id, min_user_key, max_user_key, bubble_count) VALUES (?1, ?2, ?3, ?4)",
        ) {
            Ok(s) => s,
            Err(_) => {
                let _ = cache_conn.execute_batch("ROLLBACK");
                return;
            }
        };
        for (conv_id, min_key, max_key, bubble_count) in entries {
            let _ = stmt.execute(rusqlite::params![
                conv_id,
                min_key,
                max_key,
                *bubble_count as i64
            ]);
        }
    }
    let _ = cache_conn.execute_batch("COMMIT");
}

/// Bulk-load all cached full-text entries.
fn load_cached_full_text(cache_conn: &Connection) -> HashMap<String, (String, usize)> {
    let mut map = HashMap::new();
    let mut stmt = match cache_conn
        .prepare("SELECT conv_id, full_text, bubble_count FROM conversation_full_text")
    {
        Ok(s) => s,
        Err(_) => return map,
    };
    let rows = match stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)? as usize,
        ))
    }) {
        Ok(r) => r,
        Err(_) => return map,
    };
    for row in rows.flatten() {
        map.insert(row.0, (row.1, row.2));
    }
    map
}

/// Query all bubble text for a single conversation using a prefix range scan.
/// Returns concatenated text from all bubbles (sub-millisecond per conversation).
fn query_full_text_for_conv(cursor_conn: &Connection, conv_id: &str) -> String {
    let prefix = format!("bubbleId:{}:", conv_id);
    let prefix_end = format!("bubbleId:{};", conv_id);
    let Ok(mut stmt) = cursor_conn.prepare_cached(
        "SELECT GROUP_CONCAT(NULLIF(json_extract(CAST(value AS TEXT), '$.text'), ''), ' ') \
         FROM cursorDiskKV WHERE key >= ?1 AND key < ?2",
    ) else {
        return String::new();
    };
    let result = stmt.query_row(rusqlite::params![&prefix, &prefix_end], |row| {
        row.get::<_, Option<String>>(0)
    });
    result.unwrap_or(None).unwrap_or_default()
}

/// Batch-insert full-text entries into the cache in a single transaction.
fn save_full_text_to_cache(cache_conn: &Connection, entries: &[(String, String, usize)]) {
    if entries.is_empty() {
        return;
    }
    let _ = cache_conn.execute_batch("BEGIN");
    {
        let mut stmt = match cache_conn.prepare_cached(
            "INSERT OR REPLACE INTO conversation_full_text (conv_id, full_text, bubble_count) VALUES (?1, ?2, ?3)",
        ) {
            Ok(s) => s,
            Err(_) => {
                let _ = cache_conn.execute_batch("ROLLBACK");
                return;
            }
        };
        for (conv_id, full_text, bubble_count) in entries {
            let _ = stmt.execute(rusqlite::params![conv_id, full_text, *bubble_count as i64]);
        }
    }
    let _ = cache_conn.execute_batch("COMMIT");
}

/// Delete a single conversation's rows from every sidecar cache table.
/// Best-effort: called when a conversation is deleted from Cursor's own DB so
/// the sidecar doesn't retain orphaned min/max keys or full text.
fn delete_conv_from_cache(cache_conn: &Connection, conv_id: &str) {
    let _ = cache_conn.execute(
        "DELETE FROM user_bubble_keys WHERE conv_id = ?1",
        rusqlite::params![conv_id],
    );
    let _ = cache_conn.execute(
        "DELETE FROM conversation_full_text WHERE conv_id = ?1",
        rusqlite::params![conv_id],
    );
}

/// Prune sidecar rows whose conv_id is no longer among the live conversations.
///
/// MUST only be called after the main-database enumeration succeeded: `live`
/// has to be the complete set of discovered conversation ids, otherwise valid
/// cache entries would be dropped. Nothing ever removed these rows before, so
/// the sidecar grew unbounded as conversations were deleted in Cursor.
fn prune_stale_cache_rows(cache_conn: &Connection, live: &std::collections::HashSet<&str>) {
    for table in ["user_bubble_keys", "conversation_full_text"] {
        let sql = format!("SELECT conv_id FROM {}", table);
        let existing: Vec<String> = match cache_conn.prepare(&sql) {
            Ok(mut stmt) => match stmt.query_map([], |row| row.get::<_, String>(0)) {
                Ok(rows) => rows.flatten().collect(),
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        let orphans: Vec<&String> = existing
            .iter()
            .filter(|id| !live.contains(id.as_str()))
            .collect();
        if orphans.is_empty() {
            continue;
        }
        let delete_sql = format!("DELETE FROM {} WHERE conv_id = ?1", table);
        let _ = cache_conn.execute_batch("BEGIN");
        if let Ok(mut stmt) = cache_conn.prepare_cached(&delete_sql) {
            for id in orphans {
                let _ = stmt.execute(rusqlite::params![id]);
            }
        }
        let _ = cache_conn.execute_batch("COMMIT");
    }
}

/// Build full-text map for all conversations, using cache where possible.
/// For cached conversations with matching bubble count, uses the cached text.
/// For new/changed conversations, queries per-conversation (sub-ms each) and updates cache.
fn build_full_text_map(
    cursor_conn: &Connection,
    conv_infos: &[ConvInfo],
    cache_conn: Option<&Connection>,
) -> HashMap<String, String> {
    let cached = cache_conn.map(load_cached_full_text).unwrap_or_default();
    build_full_text_map_from_cached(None, cursor_conn, conv_infos, cached, cache_conn)
}

/// Like `build_full_text_map`, but with cache entries preloaded (so the cache
/// read can happen on another thread) and, when `parallel_db_path` is given,
/// cache misses fetched in parallel across read-only connections.
///
/// Takes ownership of `cached` so cache hits can be moved (not cloned) into the
/// result map — the full-text corpus is large and used to be copied per entry.
fn build_full_text_map_from_cached(
    parallel_db_path: Option<&Path>,
    cursor_conn: &Connection,
    conv_infos: &[ConvInfo],
    mut cached: HashMap<String, (String, usize)>,
    cache_conn: Option<&Connection>,
) -> HashMap<String, String> {
    let mut full_text_map = HashMap::with_capacity(conv_infos.len());
    let mut misses: Vec<&ConvInfo> = Vec::new();

    for info in conv_infos {
        // Move the cached text out on a hit (matching bubble_count) instead of
        // cloning it. `remove` also drops stale entries we won't reuse.
        match cached.remove(&info.conv_id) {
            Some((text, count)) if count == info.bubble_count => {
                full_text_map.insert(info.conv_id.clone(), text);
            }
            // Cache miss or stale — query fresh
            _ => misses.push(info),
        }
    }

    let parallel = parallel_db_path
        .filter(|path| misses.len() > 2 && open_global_ro(path).is_some())
        .map(Path::to_path_buf);
    let new_entries: Vec<(String, String, usize)> = match parallel {
        Some(db_path) => misses
            .par_chunks(32)
            .filter_map(|chunk| {
                let conn = open_global_ro(&db_path)?;
                Some(
                    chunk
                        .iter()
                        .map(|info| {
                            (
                                info.conv_id.clone(),
                                query_full_text_for_conv(&conn, &info.conv_id),
                                info.bubble_count,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect(),
        None => misses
            .iter()
            .map(|info| {
                (
                    info.conv_id.clone(),
                    query_full_text_for_conv(cursor_conn, &info.conv_id),
                    info.bubble_count,
                )
            })
            .collect(),
    };

    // Persist fresh entries to the cache before moving them into the result map,
    // so the corpus is written straight from `new_entries` (no extra clone).
    if let Some(cache) = cache_conn {
        save_full_text_to_cache(cache, &new_entries);
    }

    for (conv_id, text, _) in new_entries {
        full_text_map.insert(conv_id, text);
    }

    full_text_map
}

/// Open an additional read-only connection to the global database so
/// independent queries can run in parallel. Returns None for databases that
/// can't be reopened by path (e.g. in-memory databases in tests).
fn open_global_ro(db_path: &Path) -> Option<Connection> {
    if !db_path.is_file() {
        return None;
    }
    Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()
}

/// Batch-fetch bubble values in parallel across read-only connections.
/// Falls back to the single supplied connection when the database can't be
/// reopened by path. Chunks that fail to fetch are skipped rather than
/// failing the whole load.
fn batch_fetch_bubbles_parallel(
    db_path: &Path,
    fallback_conn: &Connection,
    keys: &[String],
) -> Result<HashMap<String, Value>> {
    const CHUNK: usize = 200;
    if keys.len() <= CHUNK || open_global_ro(db_path).is_none() {
        return batch_fetch_bubbles(fallback_conn, keys);
    }

    let maps: Vec<HashMap<String, Value>> = keys
        .par_chunks(CHUNK)
        .filter_map(|chunk| {
            let conn = open_global_ro(db_path)?;
            batch_fetch_bubbles(&conn, chunk).ok()
        })
        .collect();

    let mut merged = HashMap::with_capacity(keys.len());
    for map in maps {
        merged.extend(map);
    }
    Ok(merged)
}

/// Batch-fetch bubble values by key using IN clauses.
fn batch_fetch_bubbles(conn: &Connection, keys: &[String]) -> Result<HashMap<String, Value>> {
    let mut bubble_map: HashMap<String, Value> = HashMap::with_capacity(keys.len());
    for chunk in keys.chunks(200) {
        let placeholders: Vec<&str> = chunk.iter().map(|_| "?").collect();
        let sql = format!(
            "SELECT key, CAST(value AS TEXT) FROM cursorDiskKV WHERE key IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> = chunk
            .iter()
            .map(|k| k as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt.query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows.flatten() {
            if let Ok(v) = serde_json::from_str::<Value>(&row.1) {
                bubble_map.insert(row.0, v);
            }
        }
    }
    Ok(bubble_map)
}

/// Load the conversation index timestamps from ItemTable.
fn load_index_timestamps(conn: &Connection) -> HashMap<String, i64> {
    let mut timestamps = HashMap::new();
    if let Ok(json) = conn.query_row::<String, _, _>(
        "SELECT value FROM ItemTable WHERE key = 'conversationClassificationScoredConversations'",
        [],
        |row| row.get(0),
    ) && let Ok(index) = serde_json::from_str::<Vec<ConversationIndexEntry>>(&json)
    {
        for entry in index {
            timestamps.insert(entry.conversation_id, entry.timestamp);
        }
    }
    timestamps
}

/// Build a Conversation from collected metadata, or None if insufficient data.
fn build_conversation(
    info: &ConvInfo,
    bubble_map: &HashMap<String, Value>,
    index_timestamps: &HashMap<String, i64>,
    workspace_map: &HashMap<String, WorkspaceInfo>,
    full_text_map: Option<&mut HashMap<String, String>>,
    db_path: &Path,
) -> Option<Conversation> {
    // Extract preview text with priority chain:
    // user_preview_key > preview_key (if user) > first_key (if user)
    let preview_text = info
        .user_preview_key
        .as_ref()
        .and_then(|k| bubble_map.get(k))
        .and_then(extract_user_text_from_value)
        .or_else(|| {
            bubble_map
                .get(&info.preview_key)
                .and_then(extract_user_text_from_value)
        })
        .or_else(|| {
            bubble_map
                .get(&info.first_key)
                .and_then(extract_user_text_from_value)
        });

    let preview_text = match preview_text {
        Some(t) if !t.is_empty() => t,
        _ => return None,
    };

    let model = bubble_map
        .get(&info.first_key)
        .and_then(|v| {
            v.get("modelType")
                .and_then(|m| m.as_str())
                .map(String::from)
        })
        .or_else(|| {
            bubble_map.get(&info.preview_key).and_then(|v| {
                v.get("modelType")
                    .and_then(|m| m.as_str())
                    .map(String::from)
            })
        });

    let ws_info = workspace_map.get(&info.conv_id);
    let workspace_path = ws_info.map(|i| i.path.clone());
    let workspace_open_path = ws_info.map(|i| i.open_path.clone());
    let title = ws_info.and_then(|i| i.title.clone());

    let project_name = workspace_path
        .as_ref()
        .map(|p| crate::history::format_short_name_from_path(p));

    // Resolve timestamp: index > first bubble createdAt > workspace composerData
    let local_ts = if let Some(&ts_millis) = index_timestamps.get(&info.conv_id) {
        let ts = Utc
            .timestamp_millis_opt(ts_millis)
            .single()
            .unwrap_or_else(Utc::now);
        ts.with_timezone(&Local)
    } else if let Some(ts) = bubble_map.get(&info.first_key).and_then(|v| {
        let created_at = v.get("createdAt")?.as_str()?;
        let dt = DateTime::parse_from_rfc3339(created_at).ok()?;
        Some(dt.with_timezone(&Local))
    }) {
        ts
    } else {
        let ws_ts = ws_info.and_then(|i| i.timestamp_millis)?;
        let ts = Utc
            .timestamp_millis_opt(ws_ts)
            .single()
            .unwrap_or_else(Utc::now);
        ts.with_timezone(&Local)
    };

    let fake_path = db_path.with_file_name(format!("cursor-{}.jsonl", info.conv_id));

    let preview = preview_text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    // All early returns above are done, so this conversation is being kept.
    // The TUI passes a full-text map (the body powers in-app search) and the
    // text is moved out of it — each conv_id is built at most once on the
    // success path — rather than cloning the whole corpus. Headless metadata
    // loads pass `None` and leave the body empty, skipping the derivation
    // entirely.
    let full_text = match full_text_map {
        Some(map) => map.remove(&info.conv_id).unwrap_or_else(|| preview.clone()),
        None => String::new(),
    };

    Some(Conversation {
        path: fake_path,
        provider: ProviderKind::Cursor,
        id: info.conv_id.clone(),
        timestamp: local_ts,
        preview,
        full_text,
        project_name,
        project_path: workspace_path,
        cwd: workspace_open_path,
        message_count: info.bubble_count,
        parse_errors: Vec::new(),
        summary: title.clone(),
        model,
        total_tokens: 0,
        duration_minutes: None,
    })
}

/// Collect the keys to fetch from a set of ConvInfo entries.
///
/// `first_key` and `preview_key` are identical unless `show_last` is set (see
/// `query_conv_infos`), so `preview_key` is only pushed when it actually
/// differs. Bubble keys embed their conversation id, so keys never collide
/// across conversations; the only remaining intra-conversation overlap
/// (`user_preview_key` equal to a boundary key) is harmless in an `IN (...)`
/// fetch, so no global sort/dedup pass is needed.
fn collect_keys_to_fetch(conv_infos: &[ConvInfo]) -> Vec<String> {
    let mut keys: Vec<String> = Vec::with_capacity(conv_infos.len() * 2);
    for info in conv_infos {
        keys.push(info.first_key.clone());
        if info.preview_key != info.first_key {
            keys.push(info.preview_key.clone());
        }
        if let Some(ref uk) = info.user_preview_key {
            keys.push(uk.clone());
        }
    }
    keys
}

/// Core logic for loading conversations from an open SQLite connection.
/// Separated from `load_from_global_db` for testability (tests use in-memory DBs).
fn load_conversations_from_conn(
    conn: &Connection,
    show_last: bool,
    include_full_text: bool,
    workspace_map: &HashMap<String, WorkspaceInfo>,
    db_path: &Path,
) -> Result<Vec<Conversation>> {
    load_conversations_from_conn_inner(
        conn,
        show_last,
        include_full_text,
        workspace_map,
        db_path,
        None,
    )
}

/// Inner implementation that accepts an optional cache connection for testability.
fn load_conversations_from_conn_inner(
    conn: &Connection,
    show_last: bool,
    include_full_text: bool,
    workspace_map: &HashMap<String, WorkspaceInfo>,
    db_path: &Path,
    cache_conn_override: Option<&Connection>,
) -> Result<Vec<Conversation>> {
    let mut conv_infos = query_conv_infos(conn, show_last)?;

    // Try cache-accelerated path; fall back to full scan on failure
    let owned_cache;
    let cache_ref = match cache_conn_override {
        Some(c) => Some(c),
        None => {
            owned_cache = open_cache_db();
            owned_cache.as_ref()
        }
    };

    if let Some(cache_conn) = cache_ref {
        let cached = load_cached_user_keys(cache_conn);
        let mut new_entries: Vec<UserKeyEntry> = Vec::new();

        for info in &mut conv_infos {
            // Cache hit only when the stored bubble_count matches the live count;
            // otherwise the conversation gained/lost messages and the cached
            // min/max user keys are stale.
            if let Some((min_key, max_key, Some(count))) = cached.get(&info.conv_id)
                && *count == info.bubble_count
            {
                info.user_preview_key = if show_last {
                    max_key.clone()
                } else {
                    min_key.clone()
                };
            } else {
                let (min_key, max_key) = query_user_key_for_conv(conn, &info.conv_id);
                info.user_preview_key = if show_last {
                    max_key.clone()
                } else {
                    min_key.clone()
                };
                new_entries.push((info.conv_id.clone(), min_key, max_key, info.bubble_count));
            }
        }

        save_user_keys_to_cache(cache_conn, &new_entries);

        // Enumeration succeeded (query_conv_infos above), so conv_infos is the
        // complete live set — safe to drop sidecar rows for conversations that
        // no longer exist.
        let live: std::collections::HashSet<&str> = conv_infos
            .iter()
            .map(|info| info.conv_id.as_str())
            .collect();
        prune_stale_cache_rows(cache_conn, &live);
    } else {
        let mut user_keys = query_user_bubble_keys(conn, show_last);
        for info in &mut conv_infos {
            info.user_preview_key = user_keys.remove(&info.conv_id);
        }
    }

    let keys_to_fetch = collect_keys_to_fetch(&conv_infos);
    let bubble_map = batch_fetch_bubbles(conn, &keys_to_fetch)?;
    let index_timestamps = load_index_timestamps(conn);
    // Headless metadata loads skip the whole full-text derivation (the corpus
    // is the largest allocation this loader makes); the TUI builds it to power
    // in-app search.
    let mut full_text_map =
        include_full_text.then(|| build_full_text_map(conn, &conv_infos, cache_ref));

    let conversations: Vec<Conversation> = conv_infos
        .iter()
        .filter_map(|info| {
            build_conversation(
                info,
                &bubble_map,
                &index_timestamps,
                workspace_map,
                full_text_map.as_mut(),
                db_path,
            )
        })
        .collect();

    Ok(conversations)
}

impl super::Provider for CursorProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Cursor
    }

    fn name(&self) -> &str {
        "Cursor (IDE)"
    }

    fn load_conversations(&self, options: LoadOptions) -> Result<Vec<Conversation>> {
        let _ = crate::debug_log::log_debug(&format!(
            "Cursor: checking global DB at {}",
            self.global_db_path.display()
        ));

        if !self.global_db_path.exists() {
            let _ = crate::debug_log::log_debug("Cursor: global DB does not exist, skipping");
            return Ok(Vec::new());
        }

        self.load_from_global_db(options.show_last, options.include_full_text)
    }

    fn load_conversations_streaming(&self, options: LoadOptions) -> Receiver<LoaderMessage> {
        let (tx, rx) = mpsc::channel();
        let global_db_path = self.global_db_path.clone();
        let workspace_storage_path = self.workspace_storage_path.clone();
        let show_last = options.show_last;
        let include_full_text = options.include_full_text;

        std::thread::spawn(move || {
            if !global_db_path.exists() {
                let _ = tx.send(LoaderMessage::Done);
                return;
            }

            let provider = CursorProvider {
                global_db_path: global_db_path.clone(),
                workspace_storage_path,
            };

            let conn = match Connection::open_with_flags(
                &global_db_path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            ) {
                Ok(c) => c,
                Err(_) => {
                    let _ = tx.send(LoaderMessage::Done);
                    return;
                }
            };

            // The workspace map (hundreds of per-workspace databases) and the
            // sidecar cache are independent of the global-database queries
            // below — load them on their own threads so their I/O overlaps
            // the scans of the (multi-GB) global database.
            let workspace_handle = std::thread::spawn(move || provider.build_workspace_map());
            let cache_handle = std::thread::spawn(move || {
                let cache_conn = open_cache_db();
                let user_keys = cache_conn
                    .as_ref()
                    .map(load_cached_user_keys)
                    .unwrap_or_default();
                // The cached full-text corpus is the largest thing this loader
                // hydrates; headless metadata loads never read it, so skip it.
                let full_text = if include_full_text {
                    cache_conn
                        .as_ref()
                        .map(load_cached_full_text)
                        .unwrap_or_default()
                } else {
                    HashMap::new()
                };
                (cache_conn, user_keys, full_text)
            });

            let index_timestamps = load_index_timestamps(&conn);

            // Enumerate all conversations via GROUP BY (index scan over every
            // bubble key; the most expensive single query in this pipeline).
            let mut conv_infos = match query_conv_infos(&conn, show_last) {
                Ok(infos) => infos,
                Err(_) => {
                    let _ = tx.send(LoaderMessage::Done);
                    return;
                }
            };

            let (cache_conn, cached_user_keys, cached_full_text) = cache_handle
                .join()
                .unwrap_or_else(|_| (None, HashMap::new(), HashMap::new()));

            // Resolve user_preview_keys from cache BEFORE splitting into phases.
            // On warm cache this fills all keys, so every conversation builds in the
            // first batch — no visible refresh when phase 2 arrives. Only trust a
            // cached entry whose bubble_count matches the live count; a stale entry
            // is left unresolved so phase 2 re-derives it from the database.
            for info in &mut conv_infos {
                if let Some((min_key, max_key, Some(count))) = cached_user_keys.get(&info.conv_id)
                    && *count == info.bubble_count
                {
                    info.user_preview_key = if show_last {
                        max_key.clone()
                    } else {
                        min_key.clone()
                    };
                }
            }

            // Enumeration succeeded above, so conv_infos is the complete live
            // set — prune sidecar rows for conversations that no longer exist.
            if let Some(ref cache) = cache_conn {
                let live: std::collections::HashSet<&str> = conv_infos
                    .iter()
                    .map(|info| info.conv_id.as_str())
                    .collect();
                prune_stale_cache_rows(cache, &live);
            }

            // Batch-fetch all keys we know about (first, preview, and cached user keys).
            let phase1_keys = collect_keys_to_fetch(&conv_infos);
            let bubble_map =
                match batch_fetch_bubbles_parallel(&global_db_path, &conn, &phase1_keys) {
                    Ok(m) => m,
                    Err(_) => {
                        let _ = tx.send(LoaderMessage::Done);
                        return;
                    }
                };

            // Build full-text search index for all conversations (cached, sub-ms
            // per miss). Headless metadata loads skip it — no corpus is built or
            // kept in memory.
            let mut full_text_map = include_full_text.then(|| {
                build_full_text_map_from_cached(
                    Some(&global_db_path),
                    &conn,
                    &conv_infos,
                    cached_full_text,
                    cache_conn.as_ref(),
                )
            });

            let workspace_map = workspace_handle.join().unwrap_or_default();

            // Phase 1: build every conversation we can (includes cached user keys).
            let mut phase1_convs = Vec::new();
            let mut remaining_infos = Vec::new();
            for info in conv_infos {
                if let Some(conv) = build_conversation(
                    &info,
                    &bubble_map,
                    &index_timestamps,
                    &workspace_map,
                    full_text_map.as_mut(),
                    &global_db_path,
                ) {
                    phase1_convs.push(conv);
                } else {
                    remaining_infos.push(info);
                }
            }

            if !phase1_convs.is_empty() {
                let _ = tx.send(LoaderMessage::Batch(phase1_convs));
            }

            // Phase 2: resolve uncached conversations (cold cache or new conversations).
            if !remaining_infos.is_empty() {
                if let Some(ref cache) = cache_conn {
                    let conv_ids: Vec<String> = remaining_infos
                        .iter()
                        .map(|info| info.conv_id.clone())
                        .collect();
                    let resolved: HashMap<String, (Option<String>, Option<String>)> =
                        query_user_keys_for_convs(&global_db_path, &conn, &conv_ids)
                            .into_iter()
                            .map(|(conv_id, min_key, max_key)| (conv_id, (min_key, max_key)))
                            .collect();

                    let mut new_entries: Vec<UserKeyEntry> = Vec::new();
                    for info in &mut remaining_infos {
                        let Some((min_key, max_key)) = resolved.get(&info.conv_id) else {
                            continue;
                        };
                        info.user_preview_key = if show_last {
                            max_key.clone()
                        } else {
                            min_key.clone()
                        };
                        new_entries.push((
                            info.conv_id.clone(),
                            min_key.clone(),
                            max_key.clone(),
                            info.bubble_count,
                        ));
                    }
                    save_user_keys_to_cache(cache, &new_entries);
                } else {
                    let mut user_keys = query_user_bubble_keys(&conn, show_last);
                    for info in &mut remaining_infos {
                        info.user_preview_key = user_keys.remove(&info.conv_id);
                    }
                }

                let extra_keys: Vec<String> = remaining_infos
                    .iter()
                    .filter_map(|info| info.user_preview_key.as_ref())
                    .filter(|k| !bubble_map.contains_key(*k))
                    .cloned()
                    .collect();

                let mut full_map = bubble_map;
                if !extra_keys.is_empty()
                    && let Ok(extra) =
                        batch_fetch_bubbles_parallel(&global_db_path, &conn, &extra_keys)
                {
                    full_map.extend(extra);
                }

                let phase2_convs: Vec<Conversation> = remaining_infos
                    .iter()
                    .filter_map(|info| {
                        build_conversation(
                            info,
                            &full_map,
                            &index_timestamps,
                            &workspace_map,
                            full_text_map.as_mut(),
                            &global_db_path,
                        )
                    })
                    .collect();

                if !phase2_convs.is_empty() {
                    let _ = tx.send(LoaderMessage::Batch(phase2_convs));
                }
            }

            let _ = tx.send(LoaderMessage::Done);
        });

        rx
    }

    fn read_entries(&self, conversation: &Conversation) -> Result<Vec<LogEntry>> {
        let bubbles = Self::load_bubbles(&conversation.id, &self.global_db_path)?;

        let entries: Vec<LogEntry> = bubbles
            .into_iter()
            .filter_map(|bubble| bubble_to_log_entry(&bubble))
            .collect();

        Ok(entries)
    }

    fn resume(&self, conversation: &Conversation, _default_args: &[String]) -> Result<()> {
        // Check if the bridge extension is installed
        if !is_extension_installed() {
            install_extension()?;
        }

        // Open/focus the workspace (reuses existing window if already open)
        // Uses cwd which stores the open_path (may be a .code-workspace file)
        if let Some(ref path) = conversation.cwd {
            Command::new("cursor")
                .arg(path.to_string_lossy().as_ref())
                .spawn()
                .map_err(|e| {
                    AppError::ClaudeExecutionError(format!("Failed to launch Cursor: {}", e))
                })?;

            // Give Cursor time to open/focus the window and activate the extension
            std::thread::sleep(std::time::Duration::from_secs(3));
        }

        // Open the conversation via URI (routes to the focused window)
        let uri = format!("cursor://{}/open?id={}", EXTENSION_ID, conversation.id);

        Command::new("open").arg(&uri).status().map_err(|e| {
            AppError::ClaudeExecutionError(format!("Failed to open Cursor URI: {}", e))
        })?;

        Ok(())
    }

    fn delete(&self, conversation: &Conversation) -> Result<()> {
        let conn = Connection::open_with_flags(
            &self.global_db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| {
            AppError::ClaudeExecutionError(format!(
                "Failed to open Cursor database for writing: {}",
                e
            ))
        })?;

        let conv_id = &conversation.id;

        // Delete all related keys from cursorDiskKV
        for prefix in &["bubbleId", "checkpointId", "codeBlockDiff"] {
            let key_prefix = format!("{}:{}:", prefix, conv_id);
            let key_prefix_end = format!("{}:{};", prefix, conv_id);
            conn.execute(
                "DELETE FROM cursorDiskKV WHERE key >= ? AND key < ?",
                rusqlite::params![&key_prefix, &key_prefix_end],
            )?;
        }

        // Delete composerData entry
        let composer_key = format!("composerData:{}", conv_id);
        conn.execute(
            "DELETE FROM cursorDiskKV WHERE key = ?",
            rusqlite::params![&composer_key],
        )?;

        // Remove from conversation index in ItemTable
        if let Ok(index_json) = conn.query_row::<String, _, _>(
            "SELECT value FROM ItemTable WHERE key = 'conversationClassificationScoredConversations'",
            [],
            |row| row.get(0),
        ) && let Ok(mut index) = serde_json::from_str::<Vec<Value>>(&index_json)
        {
            let before = index.len();
            index.retain(|e| e.get("conversationId").and_then(|v| v.as_str()) != Some(conv_id));
            if index.len() != before
                && let Ok(new_json) = serde_json::to_string(&index)
            {
                let _ = conn.execute(
                    "UPDATE ItemTable SET value = ? WHERE key = 'conversationClassificationScoredConversations'",
                    rusqlite::params![&new_json],
                );
            }
        }

        // Drop the conversation's sidecar cache rows too, so the cache doesn't
        // keep stale keys/full text for a conversation that no longer exists.
        if let Some(cache_conn) = open_cache_db() {
            delete_conv_from_cache(&cache_conn, conv_id);
        }

        Ok(())
    }
}

// --- Internal types ---

/// Parsed bubble (message) from Cursor's cursorDiskKV table.
struct Bubble {
    /// 1 = user, 2 = assistant
    bubble_type: i64,
    /// Main text content
    text: String,
    /// Lexical richText JSON (for user messages where text may be empty)
    rich_text: Option<String>,
    /// ISO 8601 timestamp
    created_at: Option<String>,
    /// Model info (e.g. "gpt-5-codex")
    model: Option<String>,
    /// Thinking block text
    thinking: Option<String>,
    /// Tool call data
    tool_name: Option<String>,
    tool_args: Option<String>,
    tool_call_id: Option<String>,
    tool_result: Option<String>,
    tool_status: Option<String>,
}

// --- Parsing helpers ---

fn parse_bubble(json_str: &str) -> Option<Bubble> {
    let v: Value = serde_json::from_str(json_str).ok()?;

    let bubble_type = v.get("type").and_then(|t| t.as_i64())?;

    let text = v
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    let rich_text = v.get("richText").and_then(|r| {
        if r.is_string() {
            r.as_str().map(String::from)
        } else if r.is_object() {
            Some(r.to_string())
        } else {
            None
        }
    });

    let created_at = v
        .get("createdAt")
        .and_then(|t| t.as_str())
        .map(String::from);

    let model = v
        .get("modelInfo")
        .and_then(|m| m.get("modelName"))
        .and_then(|n| n.as_str())
        .map(String::from);

    let thinking = v
        .get("thinking")
        .and_then(|t| t.get("text"))
        .and_then(|t| t.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let tool_former = v.get("toolFormerData");
    let tool_name = tool_former
        .and_then(|t| t.get("name"))
        .and_then(|n| n.as_str())
        .map(String::from);
    let tool_args = tool_former.and_then(|t| {
        // Try rawArgs first, fall back to params (newer Cursor format)
        let raw = t.get("rawArgs").and_then(|a| {
            if let Some(s) = a.as_str() {
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            } else if a.is_object() || a.is_array() {
                Some(a.to_string())
            } else {
                None
            }
        });
        raw.or_else(|| {
            t.get("params").and_then(|p| {
                if let Some(s) = p.as_str() {
                    if s.is_empty() {
                        None
                    } else {
                        Some(s.to_string())
                    }
                } else if p.is_object() || p.is_array() {
                    Some(p.to_string())
                } else {
                    None
                }
            })
        })
    });
    let tool_call_id = tool_former
        .and_then(|t| t.get("toolCallId"))
        .and_then(|i| i.as_str())
        .map(String::from);
    let tool_result = tool_former
        .and_then(|t| t.get("result"))
        .and_then(|r| r.as_str())
        .map(String::from);
    let tool_status = tool_former
        .and_then(|t| {
            t.get("additionalData")
                .and_then(|a| a.get("status"))
                .or_else(|| t.get("status"))
        })
        .and_then(|s| s.as_str())
        .map(String::from);

    Some(Bubble {
        bubble_type,
        text,
        rich_text,
        created_at,
        model,
        thinking,
        tool_name,
        tool_args,
        tool_call_id,
        tool_result,
        tool_status,
    })
}

/// Extract plain text from a Lexical richText JSON structure.
/// Recursively walks the tree collecting all text node values.
fn extract_text_from_richtext(rich_text_str: &str) -> Option<String> {
    let v: Value = serde_json::from_str(rich_text_str).ok()?;
    let mut parts = Vec::new();
    collect_text_nodes(&v, &mut parts);
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(""))
    }
}

fn collect_text_nodes(node: &Value, parts: &mut Vec<String>) {
    // If this node is a text node, collect its text
    if node.get("type").and_then(|t| t.as_str()) == Some("text") {
        if let Some(text) = node.get("text").and_then(|t| t.as_str()) {
            parts.push(text.to_string());
        }
        return;
    }
    // If this is a paragraph/linebreak, add a newline separator
    if node.get("type").and_then(|t| t.as_str()) == Some("linebreak") {
        parts.push("\n".to_string());
        return;
    }
    // Recurse into children
    if let Some(children) = node.get("children").and_then(|c| c.as_array()) {
        for (i, child) in children.iter().enumerate() {
            collect_text_nodes(child, parts);
            // Add newline between paragraphs
            if i < children.len() - 1
                && child.get("type").and_then(|t| t.as_str()) == Some("paragraph")
            {
                parts.push("\n".to_string());
            }
        }
    }
    // Also check "root" wrapper
    if let Some(root) = node.get("root") {
        collect_text_nodes(root, parts);
    }
}

/// Extract user text from a raw JSON bubble value.
/// Returns Some(text) only if type == BUBBLE_TYPE_USER and has non-empty text/richText.
fn extract_user_text_from_value(v: &Value) -> Option<String> {
    let btype = v.get("type").and_then(|t| t.as_i64()).unwrap_or(0);
    if btype != BUBBLE_TYPE_USER {
        return None;
    }
    let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
    if !text.is_empty() {
        return Some(text.to_string());
    }
    if let Some(rt) = v.get("richText").and_then(|r| r.as_str()) {
        return extract_text_from_richtext(rt);
    }
    None
}

/// Get the effective text content from a bubble.
fn bubble_text(bubble: &Bubble) -> String {
    if !bubble.text.is_empty() {
        return bubble.text.clone();
    }
    // Fallback to richText for user messages
    if let Some(ref rt) = bubble.rich_text
        && let Some(text) = extract_text_from_richtext(rt)
    {
        return text;
    }
    String::new()
}

/// Convert a Cursor bubble to a LogEntry for the viewer.
fn bubble_to_log_entry(bubble: &Bubble) -> Option<LogEntry> {
    let timestamp = bubble
        .created_at
        .clone()
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());

    match bubble.bubble_type {
        BUBBLE_TYPE_USER => {
            let text = bubble_text(bubble);
            if text.is_empty() {
                return None;
            }
            Some(LogEntry::User {
                message: UserMessage {
                    content: UserContent::String(text),
                },
                timestamp,
                cwd: None,
                is_meta: false,
            })
        }
        BUBBLE_TYPE_ASSISTANT => {
            let text = bubble_text(bubble);

            // Build content blocks
            let mut content_blocks: Vec<ContentBlock> = Vec::new();

            // Add thinking block whenever thinking text exists. (Cursor omits a
            // signature for many blocks; ContentBlock::Thinking has no signature
            // field anyway, so keying on the signature only hid valid thinking.)
            if let Some(thinking) = &bubble.thinking {
                content_blocks.push(ContentBlock::Thinking {
                    thinking: thinking.clone(),
                });
            }

            // Add tool use block if present (tool_call_id and tool_args may be absent)
            if let Some(name) = &bubble.tool_name {
                let id = bubble
                    .tool_call_id
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_string();
                let input: Value = bubble
                    .tool_args
                    .as_ref()
                    .and_then(|args| serde_json::from_str(args).ok())
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                content_blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });

                // Add tool result if present
                if let Some(ref result) = bubble.tool_result {
                    let truncated = truncate_str(result, 500);
                    content_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: id,
                        content: Some(Value::String(truncated)),
                        is_error: None,
                        status: bubble.tool_status.clone(),
                    });
                }
            }

            // Add text block if present
            if !text.is_empty() {
                content_blocks.push(ContentBlock::Text { text });
            }

            if content_blocks.is_empty() {
                return None;
            }

            Some(LogEntry::Assistant {
                message: AssistantMessage {
                    content: content_blocks,
                    model: bubble.model.clone(),
                    usage: None,
                    id: None,
                },
                timestamp,
            })
        }
        _ => None,
    }
}

/// Truncate a string to at most `max_bytes` bytes at a valid char boundary.
fn truncate_str(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    // Find the last valid char boundary at or before max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}

fn is_extension_installed() -> bool {
    let home = home::home_dir().unwrap_or_default();
    let extensions_dir = home.join(".cursor").join("extensions");
    let ext_dir = extensions_dir.join("mnemonai.mnemonai-bridge-0.1.0");

    // Check both: directory exists AND registered in extensions.json
    if !ext_dir.join("extension.js").exists() {
        return false;
    }

    let extensions_json_path = extensions_dir.join("extensions.json");
    if !extensions_json_path.exists() {
        return false;
    }

    let content = match std::fs::read_to_string(&extensions_json_path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    let extensions: Vec<Value> = match serde_json::from_str(&content) {
        Ok(e) => e,
        Err(_) => return false,
    };

    extensions.iter().any(|e| {
        e.get("identifier")
            .and_then(|id| id.get("id"))
            .and_then(|id| id.as_str())
            == Some(EXTENSION_ID)
    })
}

fn install_extension() -> Result<()> {
    let home = home::home_dir().unwrap_or_default();
    let extensions_dir = home.join(".cursor").join("extensions");
    let ext_dir = extensions_dir.join("mnemonai.mnemonai-bridge-0.1.0");

    // Copy extension files
    std::fs::create_dir_all(&ext_dir).map_err(AppError::Io)?;

    let package_json = include_str!("../../extension/package.json");
    std::fs::write(ext_dir.join("package.json"), package_json).map_err(AppError::Io)?;

    let extension_js = include_str!("../../extension/extension.js");
    std::fs::write(ext_dir.join("extension.js"), extension_js).map_err(AppError::Io)?;

    // Register in extensions.json so Cursor discovers it
    let extensions_json_path = extensions_dir.join("extensions.json");
    let mut extensions: Vec<Value> = if extensions_json_path.exists() {
        let content = std::fs::read_to_string(&extensions_json_path).map_err(AppError::Io)?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Check if already registered
    let already_registered = extensions.iter().any(|e| {
        e.get("identifier")
            .and_then(|id| id.get("id"))
            .and_then(|id| id.as_str())
            == Some(EXTENSION_ID)
    });

    if !already_registered {
        let entry = serde_json::json!({
            "identifier": { "id": EXTENSION_ID },
            "version": "0.1.0",
            "location": {
                "$mid": 1,
                "fsPath": ext_dir.to_string_lossy(),
                "external": format!("file://{}", ext_dir.to_string_lossy()),
                "path": ext_dir.to_string_lossy(),
                "scheme": "file"
            },
            "relativeLocation": "mnemonai.mnemonai-bridge-0.1.0"
        });
        extensions.push(entry);
        let json = serde_json::to_string(&extensions).map_err(|e| {
            AppError::ClaudeExecutionError(format!("Failed to serialize extensions.json: {}", e))
        })?;
        std::fs::write(&extensions_json_path, json).map_err(AppError::Io)?;
    }

    eprintln!("Installed mnemonai-bridge extension.");
    eprintln!("Please restart Cursor for the extension to take effect.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_richtext() {
        let rt = r#"{"root":{"children":[{"children":[{"detail":0,"format":0,"mode":"normal","style":"","text":"Hello world","type":"text","version":1}],"direction":"ltr","format":"","indent":0,"type":"paragraph","version":1}],"direction":"ltr","format":"","indent":0,"type":"root","version":1}}"#;
        assert_eq!(
            extract_text_from_richtext(rt),
            Some("Hello world".to_string())
        );
    }

    #[test]
    fn test_extract_text_from_richtext_multiline() {
        let rt = r#"{"root":{"children":[{"children":[{"text":"Line 1","type":"text"}],"type":"paragraph"},{"children":[{"text":"Line 2","type":"text"}],"type":"paragraph"}],"type":"root"}}"#;
        assert_eq!(
            extract_text_from_richtext(rt),
            Some("Line 1\nLine 2".to_string())
        );
    }

    #[test]
    fn test_parse_bubble_user_message() {
        let json = r#"{"_v":3,"type":1,"text":"Hello","createdAt":"2025-01-01T00:00:00Z","bubbleId":"test-id"}"#;
        let bubble = parse_bubble(json).unwrap();
        assert_eq!(bubble.bubble_type, 1);
        assert_eq!(bubble.text, "Hello");
        assert_eq!(bubble.created_at, Some("2025-01-01T00:00:00Z".to_string()));
    }

    #[test]
    fn test_parse_bubble_assistant_with_model() {
        let json = r#"{"_v":3,"type":2,"text":"Sure!","createdAt":"2025-01-01T00:00:01Z","bubbleId":"test-id","modelInfo":{"modelName":"claude-sonnet-4"}}"#;
        let bubble = parse_bubble(json).unwrap();
        assert_eq!(bubble.bubble_type, 2);
        assert_eq!(bubble.text, "Sure!");
        assert_eq!(bubble.model, Some("claude-sonnet-4".to_string()));
    }

    #[test]
    fn test_parse_bubble_with_thinking() {
        let json = r#"{"_v":3,"type":2,"text":"","createdAt":"2025-01-01T00:00:01Z","bubbleId":"test-id","thinking":{"text":"Let me think about this...","signature":"sig123"}}"#;
        let bubble = parse_bubble(json).unwrap();
        assert_eq!(
            bubble.thinking,
            Some("Let me think about this...".to_string())
        );
    }

    #[test]
    fn test_parse_bubble_with_tool() {
        let json = r#"{"_v":3,"type":2,"text":"","createdAt":"2025-01-01T00:00:01Z","bubbleId":"test-id","toolFormerData":{"name":"read_file","rawArgs":"{\"path\":\"test.rs\"}","toolCallId":"call_123","status":"completed","result":"file contents"}}"#;
        let bubble = parse_bubble(json).unwrap();
        assert_eq!(bubble.tool_name, Some("read_file".to_string()));
        assert_eq!(bubble.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_bubble_to_log_entry_user() {
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_USER,
            text: "Hello".to_string(),
            rich_text: None,
            created_at: Some("2025-01-01T00:00:00Z".to_string()),
            model: None,
            thinking: None,
            tool_name: None,
            tool_args: None,
            tool_call_id: None,
            tool_result: None,
            tool_status: None,
        };
        let entry = bubble_to_log_entry(&bubble).unwrap();
        match entry {
            crate::claude::LogEntry::User { message, .. } => {
                assert_eq!(crate::claude::extract_text_from_user(&message), "Hello");
            }
            _ => panic!("Expected User entry"),
        }
    }

    #[test]
    fn test_bubble_to_log_entry_assistant() {
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_ASSISTANT,
            text: "Sure, I can help!".to_string(),
            rich_text: None,
            created_at: Some("2025-01-01T00:00:01Z".to_string()),
            model: Some("claude-sonnet-4".to_string()),
            thinking: None,
            tool_name: None,
            tool_args: None,
            tool_call_id: None,
            tool_result: None,
            tool_status: None,
        };
        let entry = bubble_to_log_entry(&bubble).unwrap();
        match entry {
            crate::claude::LogEntry::Assistant { message, .. } => {
                assert_eq!(
                    crate::claude::extract_text_from_assistant(&message),
                    "Sure, I can help!"
                );
            }
            _ => panic!("Expected Assistant entry"),
        }
    }

    #[test]
    fn test_bubble_to_log_entry_empty_skipped() {
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_ASSISTANT,
            text: "".to_string(),
            rich_text: None,
            created_at: Some("2025-01-01T00:00:01Z".to_string()),
            model: None,
            thinking: None,
            tool_name: None,
            tool_args: None,
            tool_call_id: None,
            tool_result: None,
            tool_status: None,
        };
        assert!(bubble_to_log_entry(&bubble).is_none());
    }

    #[test]
    fn test_cursor_provider_loads() {
        use crate::providers::Provider;

        let provider = CursorProvider::new();
        if !provider.global_db_path.exists() {
            eprintln!("Cursor DB not found, skipping live test");
            return;
        }

        let convs = provider
            .load_conversations(LoadOptions {
                show_last: false,
                debug: None,
                include_full_text: true,
            })
            .expect("Failed to load");

        eprintln!("Loaded {} Cursor conversations", convs.len());
        assert!(!convs.is_empty(), "Expected conversations");

        let with_project = convs.iter().filter(|c| c.project_name.is_some()).count();
        let with_summary = convs.iter().filter(|c| c.summary.is_some()).count();
        eprintln!(
            "{} have project names, {} have titles",
            with_project, with_summary
        );

        for conv in convs.iter().take(3) {
            eprintln!(
                "  [{}] {} | {} | {:?}",
                &conv.id[..8],
                conv.project_name.as_deref().unwrap_or("(none)"),
                conv.summary.as_deref().unwrap_or("(no title)"),
                &conv.preview[..conv.preview.len().min(60)]
            );
        }

        // Test reading entries for first conversation
        let entries = provider
            .read_entries(&convs[0])
            .expect("Failed to read entries");
        eprintln!("First conversation has {} log entries", entries.len());
    }

    // --- In-memory SQLite tests for load_conversations_from_conn ---

    /// Create an in-memory SQLite DB with the same schema as Cursor's state.vscdb.
    fn create_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB);
             CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT);",
        )
        .unwrap();
        conn
    }

    /// Insert a bubble into the test DB.
    fn insert_bubble(conn: &Connection, conv_id: &str, bubble_uuid: &str, json: &Value) {
        let key = format!("bubbleId:{}:{}", conv_id, bubble_uuid);
        let value = serde_json::to_string(json).unwrap();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            rusqlite::params![&key, &value],
        )
        .unwrap();
    }

    /// Insert a conversation index entry so the conversation gets a timestamp.
    fn insert_index_entry(conn: &Connection, conv_id: &str, timestamp_millis: i64) {
        // Read existing index or create new
        let existing: String = conn
            .query_row(
                "SELECT value FROM ItemTable WHERE key = 'conversationClassificationScoredConversations'",
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "[]".to_string());
        let mut index: Vec<Value> = serde_json::from_str(&existing).unwrap();
        index.push(serde_json::json!({
            "conversationId": conv_id,
            "timestamp": timestamp_millis
        }));
        let new_json = serde_json::to_string(&index).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO ItemTable (key, value) VALUES ('conversationClassificationScoredConversations', ?)",
            rusqlite::params![&new_json],
        )
        .unwrap();
    }

    #[test]
    fn test_discovers_conversation_with_assistant_first_key() {
        // Core bug fix test: MIN(key) picks an assistant bubble due to random UUID ordering,
        // but the second query finds the user bubble via type=1 filter.
        let conn = create_test_db();
        let conv = "conv-aaa";

        // Insert assistant bubble with a key that sorts first (UUID starting with '0')
        insert_bubble(
            &conn,
            conv,
            "00000000-0000-0000-0000-000000000001",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "I can help with that!",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        // Insert user bubble with a key that sorts later
        insert_bubble(
            &conn,
            conv,
            "ffffffff-ffff-ffff-ffff-ffffffffffff",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "How do I fix this bug?",
                "createdAt": "2025-06-01T09:59:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(convs.len(), 1, "Should discover the conversation");
        assert_eq!(convs[0].preview, "How do I fix this bug?");
    }

    #[test]
    fn test_discovers_conversation_with_user_first_key() {
        // No regression: when MIN(key) is already a user bubble, it still works.
        let conn = create_test_db();
        let conv = "conv-bbb";

        insert_bubble(
            &conn,
            conv,
            "00000000-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Hello world",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "ffffffff-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Hi there!",
                "createdAt": "2025-06-01T10:00:01Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].preview, "Hello world");
    }

    #[test]
    fn test_skips_conversation_with_no_user_bubbles() {
        // When a conversation has only assistant bubbles, it should be skipped gracefully.
        let conn = create_test_db();
        let conv = "conv-ccc";

        insert_bubble(
            &conn,
            conv,
            "aaaa-asst1",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Starting up...",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "bbbb-asst2",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Ready to help",
                "createdAt": "2025-06-01T10:00:01Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(
            convs.len(),
            0,
            "Should skip conversation with no user bubbles"
        );
    }

    #[test]
    fn test_show_last_picks_last_user_bubble() {
        // show_last=true should use MAX(key) for user bubbles, false should use MIN(key).
        let conn = create_test_db();
        let conv = "conv-ddd";

        insert_bubble(
            &conn,
            conv,
            "00000000-user1",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "First question",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "55555555-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Answer",
                "createdAt": "2025-06-01T10:00:01Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "ffffffff-user2",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Follow-up question",
                "createdAt": "2025-06-01T10:01:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");

        let convs_first =
            load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();
        assert_eq!(convs_first[0].preview, "First question");

        let convs_last =
            load_conversations_from_conn(&conn, true, true, &ws_map, &db_path).unwrap();
        assert_eq!(convs_last[0].preview, "Follow-up question");
    }

    #[test]
    fn test_discovers_conversation_with_richtext_only() {
        // richText fallback works when text field is empty.
        let conn = create_test_db();
        let conv = "conv-eee";

        let richtext = r#"{"root":{"children":[{"children":[{"text":"Rich text content","type":"text"}],"type":"paragraph"}],"type":"root"}}"#;
        insert_bubble(
            &conn,
            conv,
            "aaaa-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "",
                "richText": richtext,
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].preview, "Rich text content");
    }

    #[test]
    fn test_multiple_conversations_mixed() {
        // Mix of discoverable and non-discoverable conversations in one DB.
        let conn = create_test_db();

        // Conversation 1: has user bubble (discoverable)
        insert_bubble(
            &conn,
            "conv-1",
            "aaaa-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Question one",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_index_entry(&conn, "conv-1", 1717236000000);

        // Conversation 2: only assistant bubbles (not discoverable)
        insert_bubble(
            &conn,
            "conv-2",
            "bbbb-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Automated response",
                "createdAt": "2025-06-01T11:00:00Z"
            }),
        );
        insert_index_entry(&conn, "conv-2", 1717239600000);

        // Conversation 3: assistant sorts first, but has user bubble (discoverable)
        insert_bubble(
            &conn,
            "conv-3",
            "00000000-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Welcome!",
                "createdAt": "2025-06-01T12:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            "conv-3",
            "zzzzzzzz-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Question three",
                "createdAt": "2025-06-01T11:59:00Z"
            }),
        );
        insert_index_entry(&conn, "conv-3", 1717243200000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(convs.len(), 2, "Should discover 2 of 3 conversations");
        let ids: Vec<&str> = convs.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"conv-1"));
        assert!(ids.contains(&"conv-3"));
        assert!(!ids.contains(&"conv-2"));
    }

    #[test]
    fn test_timestamp_from_first_key_not_user_key() {
        // Timestamp should come from the first bubble's createdAt, not the user bubble's.
        let conn = create_test_db();
        let conv = "conv-fff";

        // Assistant bubble sorts first and has the earlier timestamp
        insert_bubble(
            &conn,
            conv,
            "00000000-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Auto-greeting",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        // User bubble sorts later and has a later timestamp
        insert_bubble(
            &conn,
            conv,
            "ffffffff-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "My question",
                "createdAt": "2025-06-01T10:05:00Z"
            }),
        );
        // Don't insert an index entry — force timestamp from first bubble's createdAt

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs = load_conversations_from_conn(&conn, false, true, &ws_map, &db_path).unwrap();

        assert_eq!(convs.len(), 1);
        // The timestamp should come from the first_key (assistant at 10:00:00Z),
        // not the user_preview_key (user at 10:05:00Z).
        // Compare in UTC to avoid local timezone differences.
        let ts_utc = convs[0].timestamp.with_timezone(&Utc);
        assert_eq!(ts_utc.format("%H:%M:%S").to_string(), "10:00:00");
    }

    // --- Sidecar cache tests ---

    /// Create an in-memory cache database with the same schema as open_cache_db().
    fn create_test_cache_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE user_bubble_keys (
                 conv_id      TEXT PRIMARY KEY,
                 min_user_key TEXT,
                 max_user_key TEXT,
                 bubble_count INTEGER
             );
             CREATE TABLE conversation_full_text (
                 conv_id      TEXT PRIMARY KEY,
                 full_text    TEXT NOT NULL,
                 bubble_count INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn test_query_user_key_for_conv_finds_user() {
        let conn = create_test_db();
        let conv = "conv-cache1";

        insert_bubble(
            &conn,
            conv,
            "00000000-asst",
            &serde_json::json!({ "type": BUBBLE_TYPE_ASSISTANT, "text": "Hi" }),
        );
        insert_bubble(
            &conn,
            conv,
            "11111111-user1",
            &serde_json::json!({ "type": BUBBLE_TYPE_USER, "text": "First" }),
        );
        insert_bubble(
            &conn,
            conv,
            "eeeeeeee-user2",
            &serde_json::json!({ "type": BUBBLE_TYPE_USER, "text": "Second" }),
        );
        insert_bubble(
            &conn,
            conv,
            "ffffffff-asst2",
            &serde_json::json!({ "type": BUBBLE_TYPE_ASSISTANT, "text": "Bye" }),
        );

        let (min_key, max_key) = query_user_key_for_conv(&conn, conv);
        assert_eq!(min_key, Some(format!("bubbleId:{}:11111111-user1", conv)));
        assert_eq!(max_key, Some(format!("bubbleId:{}:eeeeeeee-user2", conv)));
    }

    #[test]
    fn test_query_user_key_for_conv_no_user() {
        let conn = create_test_db();
        let conv = "conv-cache2";

        insert_bubble(
            &conn,
            conv,
            "aaaa-asst",
            &serde_json::json!({ "type": BUBBLE_TYPE_ASSISTANT, "text": "No users here" }),
        );

        let (min_key, max_key) = query_user_key_for_conv(&conn, conv);
        assert!(min_key.is_none());
        assert!(max_key.is_none());
    }

    #[test]
    fn test_cache_round_trip() {
        let cache = create_test_cache_db();

        let entries = vec![
            (
                "conv-a".to_string(),
                Some("bubbleId:conv-a:111".to_string()),
                Some("bubbleId:conv-a:999".to_string()),
                3,
            ),
            (
                "conv-b".to_string(),
                Some("bubbleId:conv-b:222".to_string()),
                Some("bubbleId:conv-b:888".to_string()),
                5,
            ),
        ];

        save_user_keys_to_cache(&cache, &entries);
        let loaded = load_cached_user_keys(&cache);

        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded.get("conv-a"),
            Some(&(
                Some("bubbleId:conv-a:111".to_string()),
                Some("bubbleId:conv-a:999".to_string()),
                Some(3),
            ))
        );
        assert_eq!(
            loaded.get("conv-b"),
            Some(&(
                Some("bubbleId:conv-b:222".to_string()),
                Some("bubbleId:conv-b:888".to_string()),
                Some(5),
            ))
        );
    }

    #[test]
    fn test_cache_null_keys_round_trip() {
        let cache = create_test_cache_db();

        let entries = vec![("conv-empty".to_string(), None, None, 0)];

        save_user_keys_to_cache(&cache, &entries);
        let loaded = load_cached_user_keys(&cache);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("conv-empty"), Some(&(None, None, Some(0))));
    }

    #[test]
    fn test_load_with_cache_integration() {
        // End-to-end: load_conversations_from_conn_inner with a cache should
        // produce the same results and populate the cache for next time.
        let conn = create_test_db();
        let cache = create_test_cache_db();
        let conv = "conv-cache-e2e";

        insert_bubble(
            &conn,
            conv,
            "00000000-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Hello!",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "ffffffff-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "My question",
                "createdAt": "2025-06-01T10:01:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");

        // First call: populates cache
        let convs =
            load_conversations_from_conn_inner(&conn, false, true, &ws_map, &db_path, Some(&cache))
                .unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].preview, "My question");

        // Verify cache was populated
        let cached = load_cached_user_keys(&cache);
        assert!(cached.contains_key(conv));

        // Second call: uses cache (same result)
        let convs2 =
            load_conversations_from_conn_inner(&conn, false, true, &ws_map, &db_path, Some(&cache))
                .unwrap();
        assert_eq!(convs2.len(), 1);
        assert_eq!(convs2[0].preview, "My question");
    }

    #[test]
    fn test_metadata_profile_skips_full_text_but_keeps_preview() {
        // Headless metadata loads (include_full_text = false) return an empty
        // body and never derive or persist the corpus, yet still resolve the
        // preview and warm the user-key sidecar for a later --last flip.
        let conn = create_test_db();
        let cache = create_test_cache_db();
        let conv = "conv-metadata-only";

        insert_bubble(
            &conn,
            conv,
            "00000000-asst",
            &serde_json::json!({
                "type": BUBBLE_TYPE_ASSISTANT,
                "text": "Assistant reply",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_bubble(
            &conn,
            conv,
            "ffffffff-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "My question",
                "createdAt": "2025-06-01T10:01:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");

        let convs = load_conversations_from_conn_inner(
            &conn,
            false,
            false,
            &ws_map,
            &db_path,
            Some(&cache),
        )
        .unwrap();

        assert_eq!(convs.len(), 1);
        // Preview is still resolved under the metadata profile.
        assert_eq!(convs[0].preview, "My question");
        // The body is projected away — no corpus is materialized.
        assert!(convs[0].full_text.is_empty());
        // Derivation is skipped, so nothing is written to the full-text sidecar.
        assert_eq!(count_rows(&cache, "conversation_full_text", conv), 0);
        // The user-key sidecar is still warmed so a later --last load stays warm.
        assert_eq!(count_rows(&cache, "user_bubble_keys", conv), 1);
    }

    /// Seed a single user_bubble_keys row (bypassing save_user_keys_to_cache so
    /// tests can plant a deliberately stale bubble_count).
    fn seed_user_key_row(
        cache: &Connection,
        conv_id: &str,
        min_key: Option<&str>,
        max_key: Option<&str>,
        bubble_count: i64,
    ) {
        cache
            .execute(
                "INSERT OR REPLACE INTO user_bubble_keys \
                 (conv_id, min_user_key, max_user_key, bubble_count) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![conv_id, min_key, max_key, bubble_count],
            )
            .unwrap();
    }

    fn count_rows(cache: &Connection, table: &str, conv_id: &str) -> i64 {
        cache
            .query_row(
                &format!("SELECT COUNT(*) FROM {} WHERE conv_id = ?1", table),
                rusqlite::params![conv_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn test_stale_user_key_cache_is_refreshed() {
        // A conversation gained a new (last) user bubble after its user-key row
        // was cached. With staleness detection the bubble_count mismatch forces
        // a re-derive, so --show-last reflects the new last message instead of
        // the stale cached one.
        let conn = create_test_db();
        let cache = create_test_cache_db();
        let conv = "conv-stale";

        // Original user bubble (was the only user message when the row was cached).
        insert_bubble(
            &conn,
            conv,
            "11111111-user1",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Old question",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        // A newer user bubble that arrived after caching.
        insert_bubble(
            &conn,
            conv,
            "ffffffff-user2",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "New question",
                "createdAt": "2025-06-01T10:05:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        // Stale cache: bubble_count=1 (real count is 2) and max_user_key points
        // at the OLD user bubble.
        let old_key = format!("bubbleId:{}:11111111-user1", conv);
        seed_user_key_row(&cache, conv, Some(&old_key), Some(&old_key), 1);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");

        // show_last=true: the stale max_user_key would yield "Old question"; the
        // refreshed derivation must yield the true last user bubble.
        let convs =
            load_conversations_from_conn_inner(&conn, true, true, &ws_map, &db_path, Some(&cache))
                .unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].preview, "New question");

        // The cache row was rewritten with the correct count and key.
        let cached = load_cached_user_keys(&cache);
        let entry = cached.get(conv).unwrap();
        assert_eq!(entry.2, Some(2), "bubble_count should be refreshed to 2");
        assert_eq!(
            entry.1.as_deref(),
            Some(&*format!("bubbleId:{}:ffffffff-user2", conv))
        );
    }

    #[test]
    fn test_delete_conv_from_cache_removes_rows() {
        let cache = create_test_cache_db();
        let conv = "conv-del";
        let other = "conv-keep";

        seed_user_key_row(&cache, conv, Some("k1"), Some("k2"), 3);
        seed_user_key_row(&cache, other, Some("o1"), Some("o2"), 4);
        save_full_text_to_cache(
            &cache,
            &[
                (conv.to_string(), "text to delete".to_string(), 3),
                (other.to_string(), "text to keep".to_string(), 4),
            ],
        );

        delete_conv_from_cache(&cache, conv);

        assert_eq!(count_rows(&cache, "user_bubble_keys", conv), 0);
        assert_eq!(count_rows(&cache, "conversation_full_text", conv), 0);
        // The unrelated conversation's rows are untouched.
        assert_eq!(count_rows(&cache, "user_bubble_keys", other), 1);
        assert_eq!(count_rows(&cache, "conversation_full_text", other), 1);
    }

    #[test]
    fn test_prune_stale_cache_rows_drops_orphans_keeps_live() {
        let cache = create_test_cache_db();

        seed_user_key_row(&cache, "conv-live", Some("k1"), Some("k2"), 2);
        seed_user_key_row(&cache, "conv-orphan", Some("o1"), Some("o2"), 1);
        save_full_text_to_cache(
            &cache,
            &[
                ("conv-live".to_string(), "live text".to_string(), 2),
                ("conv-orphan".to_string(), "orphan text".to_string(), 1),
            ],
        );

        let mut live = std::collections::HashSet::new();
        live.insert("conv-live");
        prune_stale_cache_rows(&cache, &live);

        assert_eq!(count_rows(&cache, "user_bubble_keys", "conv-live"), 1);
        assert_eq!(count_rows(&cache, "conversation_full_text", "conv-live"), 1);
        assert_eq!(count_rows(&cache, "user_bubble_keys", "conv-orphan"), 0);
        assert_eq!(
            count_rows(&cache, "conversation_full_text", "conv-orphan"),
            0
        );
    }

    #[test]
    fn test_load_prunes_orphaned_cache_rows() {
        // End-to-end: a conversation present only in the sidecar (deleted from
        // Cursor's DB) is pruned after a successful load.
        let conn = create_test_db();
        let cache = create_test_cache_db();
        let conv = "conv-present";

        insert_bubble(
            &conn,
            conv,
            "ffffffff-user",
            &serde_json::json!({
                "type": BUBBLE_TYPE_USER,
                "text": "Still here",
                "createdAt": "2025-06-01T10:00:00Z"
            }),
        );
        insert_index_entry(&conn, conv, 1717236000000);

        // Orphan rows: no such conversation exists in `conn`.
        seed_user_key_row(&cache, "conv-gone", Some("g1"), Some("g2"), 1);
        save_full_text_to_cache(&cache, &[("conv-gone".to_string(), "ghost".to_string(), 1)]);

        let ws_map = HashMap::new();
        let db_path = PathBuf::from("/tmp/test.vscdb");
        let convs =
            load_conversations_from_conn_inner(&conn, false, true, &ws_map, &db_path, Some(&cache))
                .unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(count_rows(&cache, "user_bubble_keys", "conv-gone"), 0);
        assert_eq!(count_rows(&cache, "conversation_full_text", "conv-gone"), 0);
        // The live conversation's freshly written row survives.
        assert_eq!(count_rows(&cache, "user_bubble_keys", conv), 1);
    }

    #[test]
    fn test_bubble_to_log_entry_user_text() {
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_USER,
            text: "What does this do?".to_string(),
            rich_text: None,
            created_at: Some("2025-06-01T10:00:00Z".to_string()),
            model: None,
            thinking: None,
            tool_name: None,
            tool_args: None,
            tool_call_id: None,
            tool_result: None,
            tool_status: None,
        };
        let entry = bubble_to_log_entry(&bubble).unwrap();
        match entry {
            LogEntry::User {
                message, timestamp, ..
            } => {
                assert_eq!(timestamp, "2025-06-01T10:00:00Z");
                assert_eq!(
                    crate::claude::extract_text_from_user(&message),
                    "What does this do?"
                );
            }
            _ => panic!("Expected User entry"),
        }
    }

    #[test]
    fn test_bubble_to_log_entry_assistant_thinking_without_signature() {
        // Regression: thinking blocks used to be emitted only when a signature
        // was present. Now unsigned thinking is kept.
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_ASSISTANT,
            text: "Here is the answer".to_string(),
            rich_text: None,
            created_at: Some("2025-06-01T10:00:01Z".to_string()),
            model: Some("claude-sonnet-4".to_string()),
            thinking: Some("Considering the options".to_string()),
            tool_name: None,
            tool_args: None,
            tool_call_id: None,
            tool_result: None,
            tool_status: None,
        };
        let entry = bubble_to_log_entry(&bubble).unwrap();
        match entry {
            LogEntry::Assistant { message, .. } => {
                assert_eq!(message.model.as_deref(), Some("claude-sonnet-4"));
                let has_thinking = message.content.iter().any(|b| {
                    matches!(b, ContentBlock::Thinking { thinking } if thinking == "Considering the options")
                });
                assert!(has_thinking, "unsigned thinking block should be present");
                assert_eq!(
                    crate::claude::extract_text_from_assistant(&message),
                    "Here is the answer"
                );
            }
            _ => panic!("Expected Assistant entry"),
        }
    }

    #[test]
    fn test_bubble_to_log_entry_assistant_tool_use_and_result() {
        let bubble = Bubble {
            bubble_type: BUBBLE_TYPE_ASSISTANT,
            text: String::new(),
            rich_text: None,
            created_at: Some("2025-06-01T10:00:02Z".to_string()),
            model: None,
            thinking: None,
            tool_name: Some("read_file".to_string()),
            tool_args: Some(r#"{"path":"main.rs"}"#.to_string()),
            tool_call_id: Some("call_42".to_string()),
            tool_result: Some("fn main() {}".to_string()),
            tool_status: Some("completed".to_string()),
        };
        let entry = bubble_to_log_entry(&bubble).unwrap();
        let LogEntry::Assistant { message, .. } = entry else {
            panic!("Expected Assistant entry");
        };

        let tool_use = message.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
            _ => None,
        });
        let (id, name, input) = tool_use.expect("tool_use block");
        assert_eq!(id, "call_42");
        assert_eq!(name, "read_file");
        assert_eq!(input.get("path").and_then(|v| v.as_str()), Some("main.rs"));

        let tool_result = message.content.iter().find_map(|b| match b {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                status,
                ..
            } => Some((tool_use_id, content, status)),
            _ => None,
        });
        let (result_id, content, status) = tool_result.expect("tool_result block");
        assert_eq!(result_id, "call_42");
        assert_eq!(
            content.as_ref().and_then(|v| v.as_str()),
            Some("fn main() {}")
        );
        assert_eq!(status.as_deref(), Some("completed"));
    }
}
