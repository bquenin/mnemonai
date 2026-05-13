//! Persistent conversation metadata and search-text index.
//!
//! File-backed providers use this sidecar database to skip reparsing unchanged
//! JSONL transcripts and to reuse lowercased search text on warm starts.

use crate::history::{Conversation, ProviderKind};
use crate::tui::search::{normalize_for_search, topic_end_for_text};
use chrono::{DateTime, Local};
use rusqlite::{Connection, params};
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
    pub fn conversation_if_fresh(&self, fingerprint: SourceFingerprint) -> Option<Conversation> {
        (self.fingerprint == fingerprint).then(|| self.conversation.clone())
    }
}

pub fn fingerprint_from_metadata(metadata: &Metadata) -> SourceFingerprint {
    SourceFingerprint {
        modified_millis: system_time_to_millis(metadata.modified().unwrap_or(UNIX_EPOCH)),
        size: metadata.len().min(i64::MAX as u64) as i64,
    }
}

pub fn attach_search_cache(conversation: &mut Conversation) {
    if conversation.search_text_lower.is_none() {
        conversation.search_text_lower = Some(normalize_for_search(&conversation.full_text));
    }

    if conversation.search_topic_end.is_none()
        && let Some(text_lower) = &conversation.search_text_lower
    {
        conversation.search_topic_end = Some(topic_end_for_text(text_lower));
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
    let mut iter = entries.into_iter().peekable();
    if iter.peek().is_none() {
        return;
    }

    let Some(mut conn) = open_index_db() else {
        return;
    };

    let tx = match conn.transaction() {
        Ok(tx) => tx,
        Err(_) => return,
    };

    {
        let mut stmt = match tx.prepare(
            "INSERT OR REPLACE INTO file_conversations (
                schema_version, provider, show_last, source_path, source_mtime_millis,
                source_size, id, timestamp, preview, full_text, search_text_lower,
                search_topic_end, project_name, project_path, cwd, message_count,
                parse_errors_json, summary, model, total_tokens, duration_minutes
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                ?15, ?16, ?17, ?18, ?19, ?20, ?21
            )",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return,
        };

        let provider_key = provider_key(&provider);
        let show_last = i64::from(show_last);

        for (conversation, fingerprint) in iter {
            let Some(text_lower) = &conversation.search_text_lower else {
                continue;
            };
            let topic_end = conversation
                .search_topic_end
                .unwrap_or_else(|| topic_end_for_text(text_lower));
            let parse_errors_json = serde_json::to_string(&conversation.parse_errors)
                .unwrap_or_else(|_| "[]".to_string());

            if stmt
                .execute(params![
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
                    text_lower,
                    topic_end as i64,
                    conversation.project_name.as_deref(),
                    path_to_string(conversation.project_path.as_ref()),
                    path_to_string(conversation.cwd.as_ref()),
                    conversation.message_count as i64,
                    parse_errors_json,
                    conversation.summary.as_deref(),
                    conversation.model.as_deref(),
                    conversation.total_tokens.min(i64::MAX as u64) as i64,
                    conversation
                        .duration_minutes
                        .map(|v| v.min(i64::MAX as u64) as i64),
                ])
                .is_err()
            {
                return;
            }
        }
    }

    let _ = tx.commit();
}

pub fn delete_conversation(provider: ProviderKind, path: &std::path::Path) {
    let Some(conn) = open_index_db() else {
        return;
    };

    let _ = conn.execute(
        "DELETE FROM file_conversations WHERE provider = ?1 AND source_path = ?2",
        params![provider_key(&provider), path.to_string_lossy()],
    );
}

fn load_provider_cache_from_conn(
    conn: &Connection,
    provider: ProviderKind,
    show_last: bool,
) -> rusqlite::Result<HashMap<PathBuf, CachedFileConversation>> {
    let mut stmt = conn.prepare(
        "SELECT
            source_path, source_mtime_millis, source_size, id, timestamp, preview,
            full_text, search_text_lower, search_topic_end, project_name, project_path,
            cwd, message_count, parse_errors_json, summary, model, total_tokens,
            duration_minutes
         FROM file_conversations
         WHERE schema_version = ?1 AND provider = ?2 AND show_last = ?3",
    )?;

    let rows = stmt.query_map(
        params![
            SCHEMA_VERSION,
            provider_key(&provider),
            i64::from(show_last)
        ],
        |row| {
            let source_path: String = row.get(0)?;
            let timestamp: String = row.get(4)?;
            let timestamp = DateTime::parse_from_rfc3339(&timestamp)
                .map(|ts| ts.with_timezone(&Local))
                .unwrap_or_else(|_| Local::now());
            let search_topic_end: i64 = row.get(8)?;
            let message_count: i64 = row.get(12)?;
            let parse_errors_json: String = row.get(13)?;
            let parse_errors = if parse_errors_json == "[]" {
                Vec::new()
            } else {
                serde_json::from_str(&parse_errors_json).unwrap_or_default()
            };
            let total_tokens: i64 = row.get(16)?;
            let duration_minutes: Option<i64> = row.get(17)?;

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
                        search_text_lower: Some(row.get(7)?),
                        search_topic_end: Some(search_topic_end.max(0) as usize),
                        project_name: row.get(9)?,
                        project_path: optional_path(row.get(10)?),
                        cwd: optional_path(row.get(11)?),
                        message_count: message_count.max(0) as usize,
                        parse_errors,
                        summary: row.get(14)?,
                        model: row.get(15)?,
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
    let conn = Connection::open(dir.join("conversation_index.db")).ok()?;
    init_schema(&conn).ok()?;
    Some(conn)
}

fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE IF NOT EXISTS file_conversations (
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
             search_text_lower     TEXT NOT NULL,
             search_topic_end      INTEGER NOT NULL,
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
         );
         CREATE INDEX IF NOT EXISTS idx_provider_source_path
         ON file_conversations (provider, source_path);",
    )
}

fn provider_key(provider: &ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Claude => "claude",
        ProviderKind::Cursor => "cursor",
        ProviderKind::CursorAgent => "cursor-agent",
    }
}

fn system_time_to_millis(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

fn path_to_string(path: Option<&PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().to_string())
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
    fn attach_search_cache_populates_lowercase_text() {
        let mut conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));

        attach_search_cache(&mut conversation);

        assert_eq!(
            conversation.search_text_lower.as_deref(),
            Some("hello cache body")
        );
        assert_eq!(
            conversation.search_topic_end,
            Some("hello cache body".len())
        );
    }

    #[test]
    fn load_cache_round_trips_conversation() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let mut conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        conversation.parse_errors.push(crate::history::ParseError {
            line_number: 7,
            line_content: "{bad json".to_string(),
            error_message: "expected value".to_string(),
            context_before: vec!["before".to_string()],
            context_after: vec!["after".to_string()],
        });
        attach_search_cache(&mut conversation);
        let fingerprint = SourceFingerprint {
            modified_millis: 123,
            size: 456,
        };

        save_conversations_to_conn(
            &conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        let cache = load_provider_cache_from_conn(&conn, ProviderKind::Claude, false).unwrap();
        let cached = cache.get(&conversation.path).unwrap();
        let loaded = cached.conversation_if_fresh(fingerprint).unwrap();

        assert_eq!(loaded.id, "session-1");
        assert_eq!(loaded.parse_errors.len(), 1);
        assert_eq!(loaded.parse_errors[0].line_number, 7);
        assert_eq!(
            loaded.search_text_lower.as_deref(),
            Some("hello cache body")
        );
        assert!(
            cached
                .conversation_if_fresh(SourceFingerprint {
                    modified_millis: 999,
                    size: 456
                })
                .is_none()
        );
    }

    #[test]
    fn delete_conversation_removes_cached_rows_for_all_preview_modes() {
        let mut conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let mut conversation = make_conversation(PathBuf::from("/tmp/session.jsonl"));
        attach_search_cache(&mut conversation);
        let fingerprint = SourceFingerprint {
            modified_millis: 123,
            size: 456,
        };

        save_conversations_to_conn(
            &conn,
            ProviderKind::Claude,
            false,
            [(&conversation, fingerprint)],
        )
        .unwrap();
        save_conversations_to_conn(
            &conn,
            ProviderKind::Claude,
            true,
            [(&conversation, fingerprint)],
        )
        .unwrap();

        delete_conversation_from_conn(&mut conn, ProviderKind::Claude, &conversation.path).unwrap();

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

    fn save_conversations_to_conn<'a, I>(
        conn: &Connection,
        provider: ProviderKind,
        show_last: bool,
        entries: I,
    ) -> rusqlite::Result<()>
    where
        I: IntoIterator<Item = (&'a Conversation, SourceFingerprint)>,
    {
        let mut stmt = conn.prepare(
            "INSERT OR REPLACE INTO file_conversations (
                schema_version, provider, show_last, source_path, source_mtime_millis,
                source_size, id, timestamp, preview, full_text, search_text_lower,
                search_topic_end, project_name, project_path, cwd, message_count,
                parse_errors_json, summary, model, total_tokens, duration_minutes
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                ?15, ?16, ?17, ?18, ?19, ?20, ?21
            )",
        )?;

        for (conversation, fingerprint) in entries {
            let parse_errors_json = serde_json::to_string(&conversation.parse_errors).unwrap();
            stmt.execute(params![
                SCHEMA_VERSION,
                provider_key(&provider),
                i64::from(show_last),
                conversation.path.to_string_lossy(),
                fingerprint.modified_millis,
                fingerprint.size,
                conversation.id,
                conversation.timestamp.to_rfc3339(),
                conversation.preview,
                conversation.full_text,
                conversation.search_text_lower.as_deref().unwrap(),
                conversation.search_topic_end.unwrap() as i64,
                conversation.project_name.as_deref(),
                path_to_string(conversation.project_path.as_ref()),
                path_to_string(conversation.cwd.as_ref()),
                conversation.message_count as i64,
                parse_errors_json,
                conversation.summary.as_deref(),
                conversation.model.as_deref(),
                conversation.total_tokens as i64,
                conversation.duration_minutes.map(|v| v as i64),
            ])?;
        }

        Ok(())
    }

    fn delete_conversation_from_conn(
        conn: &mut Connection,
        provider: ProviderKind,
        path: &std::path::Path,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "DELETE FROM file_conversations WHERE provider = ?1 AND source_path = ?2",
            params![provider_key(&provider), path.to_string_lossy()],
        )?;
        Ok(())
    }
}
