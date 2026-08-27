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

# Search conversation content, ranked, with match snippets (JSON array)
mnemonai search reusable workflow secrets --since 90d --limit 5 --json

# Grep within one conversation's messages, keeping 1 neighbor per match
mnemonai show <session-id-or-path> --grep job_workflow_ref --context 1 --json

# Skill-style pipeline
mnemonai list --jsonl --limit 500 | jq -r 'select(.message_count > 50) | .id'
```

`list` and `show` accept `--provider <claude|codex|cursor|cursor-agent>`,
`--local` (current directory tree), and `--show-deleted-projects`. `list` also
takes `--limit <n>`, `--since <duration>` (for example `7d`, `24h`, `2w`),
`--after <timestamp>`, `--before <timestamp>`, and `--cwd <path>`.
`--after`/`--before` accept RFC 3339 timestamps or `YYYY-MM-DD`; the lower bound
is inclusive and the upper bound is exclusive. `--cwd` matches conversations
whose recorded `cwd` or `project_path` is at or under the path. Output is JSON by
default; on errors (no match, ambiguous target) the command prints a message to
stderr and exits non-zero.

`search <words>...` ranks conversations by content, finding sessions that a
preview-only `list` scan misses. The positional words are joined with single
spaces into one query and matched with the same ranking as the interactive TUI
type-ahead: every word must appear (AND), matching is case-insensitive, and each
word matches as a substring anywhere in the text, never whole-word — so
`search job flow` also matches "jobs" and "workflows". Ranking saturates and
balances each term's frequency, favors standalone occurrences of one- and
two-character terms, and boosts conversations where all terms occur nearby;
this prevents one noisy substring from dominating a multi-term query. It
accepts `--provider`, `--local` / `--cwd`, and
`--since` / `--after` / `--before` with the same meaning as `list`, plus
`--limit <n>` (default 10), `--snippets <0..=5>` (default 2), and a repeatable
`--exclude-session <id>` that drops specific conversations (handy for excluding
the live session). Output is a JSON array by default; pass `--jsonl` for one
object per line. Snippets are ~240-character windows sliced from the lowercased
conversation text. Multi-term searches lead with a nearby span containing every
term when available; single-term searches retain earliest-match ordering.

`show` additionally accepts a repeatable `--grep <pattern>` (case-insensitive
substring, OR across patterns; matches a message's text, thinking, and
stringified tool input/result) and `--context <n>` (default 1). With `--grep`,
the output keeps only matching messages plus `n` neighbors on each side (merging
overlapping ranges), flags each real match with `"matched": true`, and adds a
`total_messages` field with the pre-filter count. Without `--grep`, `show`
output is unchanged.

Scope for `search` and `--grep` defaults to global; pass `--local` or `--cwd`, or
place the global `--global` flag before the subcommand
(`mnemonai --global search <words>`).

Headless output depends only on these flags, never on the config file: unlike the
interactive TUI, `list`/`show` ignore the config's `local`, `exclude`, and
`show_deleted_projects` settings so scripts and skills get the same result on any
machine. Pass `--local` / `--show-deleted-projects` explicitly when you want them.

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

### `search` output — ranked matches

Each result is a slim object ordered by `score` descending (then `timestamp`
descending). No `preview` or `parse_errors`; use `show` for full detail.

| Field | Type | Notes |
|-------|------|-------|
| `provider` | string | `claude` \| `codex` \| `cursor` \| `cursor-agent` |
| `id` | string | session ID (use with `show`) |
| `path` | string | absolute source path |
| `timestamp` | string | RFC 3339 |
| `project_name` | string? | |
| `cwd` | string? | |
| `summary` | string? | title, when available |
| `score` | number | relevance (match density + topic-window boost + recency) |
| `match_count` | number | total query-term occurrences in the conversation |
| `snippets` | array | up to `--snippets` lowercased ~240-char context windows |

### `show` output — conversation detail

`{ "conversation": <summary>, "messages": [<message>, ...] }` (with `--grep`, a
`total_messages` count is inserted before `messages`), where each message carries
`role` plus whichever of these apply (absent fields are omitted):

| Field | Populated for |
|-------|---------------|
| `index` | always — zero-based normalized message index |
| `entry_index` | always — zero-based source log entry index |
| `block_index` | content-block messages, when applicable |
| `role` | always — one of `summary`, `user`, `assistant`, `tool_call`, `tool_result`, `thinking`, `image`, `system`, or `agent_<type>` for sub-agent turns |
| `matched` | `--grep` matches only — `true` on messages that matched a pattern (context neighbors omit it) |
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

## Claude Code Integration

Optional session-continuity tooling for Claude Code, built on `mnemonai show`.
Long sessions get monotonically more expensive: every turn re-reads the whole
accumulated context, so a 400K-token session costs ~5x per turn what a fresh
one does. This integration makes fresh-session resumes cheap enough to prefer:

- **`handoff` skill** (`skills/handoff/`) — say `resume session <id>` in a
  fresh session and it distills the old transcript into a compact handoff file
  (`~/.claude/handoffs/<workstream>.md`), then continues from there. Works
  post-hoc: the old session never has to prepare anything. Also supports
  explicit end-of-day checkpoints (`handoff`).
- **`scripts/session-digest.sh`** — the distiller: user prompts + assistant
  prose only (no tool output/thinking), recent messages kept fuller, capped at
  ~30K tokens. Provider-agnostic via `mnemonai show --json`; accepts id
  prefixes.
- **`hooks/context-warn.sh`** (UserPromptSubmit) — one-line warning when the
  session crosses 200K context tokens (then each further 100K), with the
  estimated $/turn for the current model and the cheap-resume phrase
  (`resume session <id>`) for later.

Install (symlinks into `~/.claude`, so `git pull` updates them):

```bash
./scripts/install-claude-integration.sh
```

Then merge the printed hook block into `~/.claude/settings.json`. Warning
threshold is tunable via `MNEMONAI_CTX_WARN_TOKENS` (default 200000).

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
