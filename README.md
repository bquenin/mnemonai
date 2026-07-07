# mnemonai

Universal AI coding conversation history browser. Search, browse, and resume conversations across multiple AI coding tools from a single TUI.

## Supported Tools

| Tool | History Format | Resume Support |
|------|---------------|----------------|
| **Claude Code** | JSONL files in `~/.claude/projects/` | `claude --resume <session-id>` |
| **Codex** | JSONL transcripts in `$CODEX_HOME/sessions/` (default `~/.codex/sessions/`) | `codex resume <session-id>` |
| **Cursor Agent CLI** | JSONL transcripts in `~/.cursor/projects/*/agent-transcripts/` | `agent --resume <chat-id>` |
| **Cursor (IDE)** | SQLite in workspace storage | Bridge extension + `cursor://` URI |

## Supported Platforms

- macOS (Apple Silicon and Intel)
- Linux (x86_64)

## Install

### Homebrew

```bash
brew install bquenin/mnemonai/mnemonai
```

### Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/bquenin/mnemonai/main/scripts/install.sh | bash
```

## Usage

```bash
# Launch the TUI scoped to the current directory tree
mnemonai

# Show all conversations across all providers
mnemonai --global

# Force current-directory-tree scope even when config sets local = false
mnemonai --local
```

## Headless Usage

Stable, non-interactive commands for tools and skills that want to analyze
structured conversation data instead of scraping the TUI.

```bash
# List all conversations as a JSON array
mnemonai list --json

# Stream conversation summaries as JSONL (one object per line)
mnemonai list --jsonl --provider codex --limit 100

# Scope list output for retrospective analysis
mnemonai list --json --since 7d --limit 50
mnemonai list --json --cwd ~/code/my-repo --after 2026-06-01 --before 2026-06-20

# Show one conversation by session ID or source path
mnemonai show <session-id-or-path> --json

# Search conversations with the same ranking used by the TUI
mnemonai search "emp tower SP cnc3" --json --limit 20 --since 180d

# Extract deterministic message windows for a session
mnemonai excerpt <session-id-or-path> --query "emp tower SP" --context 3 --json
mnemonai excerpt <session-id-or-path> --query "emp tower SP" --match-roles user,assistant,summary --max-windows 3 --json
mnemonai excerpt <session-id-or-path> --from 24 --to 39 --markdown

# Skill-style pipeline
mnemonai search "failed tests" --jsonl --limit 20 | jq -r '.id'
```

`list`, `show`, `search`, and `excerpt` accept
`--provider <claude|codex|cursor|cursor-agent>`, `--local` (current directory
tree), and `--show-deleted-projects`. `list` and `search` also take
`--limit <n>`, `--since <duration>` (for example `7d`, `24h`, `2w`),
`--after <timestamp>`, `--before <timestamp>`, and `--cwd <path>`.
`--after`/`--before` accept RFC 3339 timestamps or `YYYY-MM-DD`; the lower bound
is inclusive and the upper bound is exclusive. `--cwd` matches conversations
whose recorded `cwd` or `project_path` is at or under the path. Output is JSON by
default except for `excerpt --markdown`; on errors (no match, ambiguous target,
invalid range) the command prints a message to stderr and exits non-zero.

Headless output depends only on these flags, never on the config file: unlike the
interactive TUI, headless commands ignore the config's `local`, `exclude`, and
`show_deleted_projects` settings so scripts and skills get the same result on
any machine. Pass `--local` / `--show-deleted-projects` explicitly when you want
them.

### `list` output — conversation summary

Each conversation is an object with these fields (nullable fields are omitted
when empty):

| Field | Type | Notes |
|-------|------|-------|
| `provider` | string | `claude` \| `codex` \| `cursor` \| `cursor-agent` |
| `id` | string | session ID (use with `show`) |
| `path` | string | absolute source path |
| `timestamp` | string | RFC 3339 |
| `project_name` | string? | |
| `project_path` | string? | |
| `cwd` | string? | |
| `preview` | string | short text preview |
| `summary` | string? | title, when available |
| `model` | string? | when available |
| `message_count` | number | |
| `total_tokens` | number | |
| `duration_minutes` | number? | |
| `parse_errors` | array | diagnostics; entries can include raw transcript lines, so this may be large for broken transcripts |

### `search` output — ranked conversation summary

`search` returns the same conversation summary fields as `list`, plus:

| Field | Type | Notes |
|-------|------|-------|
| `score` | number | TUI-derived relevance score; compare only within one search result set |

### `show` output — conversation detail

`{ "conversation": <summary>, "messages": [<message>, ...] }`, where each message
carries `role` plus whichever of these apply (absent fields are omitted):

| Field | Populated for |
|-------|---------------|
| `index` | always — zero-based normalized message index |
| `entry_index` | always — zero-based source log entry index |
| `block_index` | content-block messages, when applicable |
| `role` | always — one of `summary`, `user`, `assistant`, `tool_call`, `tool_result`, `thinking`, `image`, `system`, or `agent_<type>` for sub-agent turns |
| `timestamp` | most messages (RFC 3339) |
| `text` | text messages, tool results, summaries |
| `tool_call_id` | `tool_call` and `tool_result`; use this to pair calls with results |
| `tool_name` | `tool_call` |
| `tool_input` | `tool_call` — raw, tool-specific JSON (opaque) |
| `tool_result` | `tool_result` — raw, provider-specific JSON (opaque) |
| `tool_result_status` | `tool_result`, when the provider exposes status |
| `tool_result_exit_code` | `tool_result`, when command-style output exposes an exit code |
| `tool_result_error` | `tool_result`, when an explicit or recognizable error/success marker exists |
| `thinking` | `thinking` |
| `model` | assistant/thinking, when available |
| `agent_id` | sub-agent messages |
| `subtype` / `level` / `duration_ms` | `system` |
| `source` | `image` — raw provider-specific image source (opaque) |

`tool_call_id`, `index`, `entry_index`, and `block_index` are intended for
tool-trace analysis. `tool_input`, `tool_result`, and `source` are passed
through verbatim from the underlying transcript; treat their inner shape as
opaque rather than a stable contract.

### `excerpt` output — conversation windows

`excerpt` returns:

```json
{
  "conversation": "<summary object>",
  "windows": [
    {
      "start_index": 24,
      "end_index": 39,
      "messages": ["<message objects from show output>"]
    }
  ]
}
```

Use `--query <text>` to find message windows around matching terms, or
`--from <index> [--to <index>]` to select an explicit inclusive range. Query
excerpting searches normalized message text, thinking text, tool names, and tool
inputs, then merges overlapping windows. Use `--match-roles <roles>` to limit
which message roles can trigger query matches while still returning surrounding
context messages, and `--max-windows <n>` to cap the number of returned windows.

### Agent memory retrieval

For skills and agent harnesses, keep `mnemonai` as the deterministic retrieval
backend and let the current agent do the reasoning:

```bash
mnemonai search "topic words" --json --cwd . --limit 10
mnemonai excerpt "$session_id" --query "topic words" --context 3 --match-roles user,assistant,summary --max-windows 3 --json
```

Agents should summarize only the retrieved excerpts that matter to the current
task and cite provider, session ID/path, and message ranges.

This repository includes a starter skill at `skills/mnemonai-memory` with the
recommended agent workflow for searching, excerpting, and citing prior sessions.

## Keyboard Shortcuts

Press `?` at any time to open the help overlay.

### List View

| Key | Action |
|-----|--------|
| Type | Fuzzy search conversations |
| `Up/Down` | Navigate list |
| `Home/End` | Jump to first/last |
| `Page Up/Down` | Page navigation |
| `Ctrl+D/U` | Half page down/up |
| `Ctrl+N/P` | Next/prev (emacs-style) |
| `Enter` | View conversation |
| `Ctrl+R` | Resume conversation in original tool |
| `Ctrl+X` | Delete conversation |
| `Ctrl+O` | Select and exit |
| `Ctrl+W` | Delete word |
| `Esc` | Quit |

### Detail View

| Key | Action |
|-----|--------|
| `Up/Down` or `j/k` | Scroll |
| `d/u` or `Ctrl+D/U` | Half page down/up |
| `Page Up/Down` | Page navigation |
| `g/G` or `Home/End` | Jump to top/bottom |
| `/` | Search within conversation |
| `n/N` | Next/prev search match |
| `t` | Toggle tool display |
| `T` | Toggle thinking blocks |
| `i` | Toggle timestamps and durations |
| `p` | Show file path |
| `Y` | Copy path to clipboard |
| `I` | Copy session ID to clipboard |
| `e` | Export to file |
| `y` | Copy to clipboard |
| `Ctrl+R` | Resume conversation |
| `Ctrl+X` | Delete conversation |
| `Esc` or `q` | Back to list |

## Configuration

Create `~/.config/mnemonai/config.toml`:

```toml
# Scope interactive startup to the current directory tree.
# Unset defaults to true; set false to start globally.
local = true

# Hide projects whose name contains any of these strings
exclude = ["some-project", "another-project"]

[display]
no_tools = false              # Hide tool-use messages
relative_time = true          # "2 hours ago" vs "2026-02-18 14:30"
last = false                  # Show last messages in preview (vs first)
show_thinking = false         # Show thinking blocks
plain = false                 # Plain text output without formatting
pager = false                 # Use pager (less) for output
show_deleted_projects = false # Include conversations from deleted directories

[resume]
default_args = []             # Default args passed to 'claude --resume' for Claude Code sessions
```

## Releasing

1. Bump the version in `Cargo.toml`
2. Commit and push
3. Tag and push the tag:

```bash
git tag v0.x.x
git push origin v0.x.x
```

The GitHub Actions release workflow triggers on `v*` tags, builds platform binaries, creates the GitHub release, and updates the Homebrew tap.

## Acknowledgments

mnemonai is a fork of [claude-history](https://github.com/raine/claude-history) by [Raine Virta](https://github.com/raine).

## License

MIT
