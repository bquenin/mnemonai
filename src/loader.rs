//! Shared conversation-loading helpers used by both the interactive TUI and the
//! headless commands so they apply identical local and provider filtering.

use crate::cli::{DebugLevel, ProviderFilter};
use crate::error::{AppError, Result};
use crate::history::{Conversation, ProviderKind};
use crate::providers::Provider;

/// Whether a provider should be included given an optional provider filter.
/// `None` matches every provider.
pub fn provider_filter_matches(filter: Option<ProviderFilter>, kind: &ProviderKind) -> bool {
    filter.is_none_or(|filter| &filter.kind() == kind)
}

/// Load conversations scoped to the current directory across all providers.
///
/// Mirrors the TUI's local-mode loading: non-Claude providers are filtered to
/// the current directory, while Claude's loader is already directory-scoped.
/// Providers that fail to load are silently skipped. The result is unsorted and
/// unindexed; callers sort by timestamp and assign display indexes.
pub fn load_local(
    providers: &[Box<dyn Provider>],
    show_last: bool,
    debug: Option<DebugLevel>,
    provider_filter: Option<ProviderFilter>,
) -> Result<Vec<Conversation>> {
    let current_dir = std::env::current_dir().map_err(|err| {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Failed to get current directory: {}", err),
        ))
    })?;

    let mut conversations = Vec::new();
    for provider in providers {
        if !provider_filter_matches(provider_filter, &provider.kind()) {
            continue;
        }

        if let Ok(mut provider_conversations) = provider.load_conversations(show_last, debug) {
            // For non-Claude providers, filter to the current directory.
            if provider.kind() != ProviderKind::Claude {
                provider_conversations.retain(|conversation| {
                    conversation
                        .project_path
                        .as_ref()
                        .is_some_and(|path| path == &current_dir)
                });
            }
            conversations.extend(provider_conversations);
        }
    }

    Ok(conversations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_filter_matches_expected_kinds() {
        assert!(provider_filter_matches(None, &ProviderKind::Cursor));
        assert!(provider_filter_matches(
            Some(ProviderFilter::Codex),
            &ProviderKind::Codex
        ));
        assert!(!provider_filter_matches(
            Some(ProviderFilter::Codex),
            &ProviderKind::Claude
        ));
        assert!(provider_filter_matches(
            Some(ProviderFilter::CursorAgent),
            &ProviderKind::CursorAgent
        ));
    }
}
