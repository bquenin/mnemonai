# Agent Memory Retrieval Plan

## Goal

Make `mnemonai` useful as a model-free retrieval backend for agent skills and
agent harnesses. `mnemonai` should find, identify, and excerpt prior sessions;
the current agent should do all interpretation, summarization, and application
of the retrieved context.

The target workflow is:

```text
user asks for past context -> agent calls mnemonai -> mnemonai returns relevant
sessions/excerpts with stable source identifiers -> agent summarizes and applies
the context in the current session
```

## Non-Goals

- Do not add an LLM, embedding service, or summarizer to `mnemonai`.
- Do not make provider-specific session resume the central abstraction.
- Do not require agents to scrape TUI output or provider-specific transcript
  stores.
- Do not inject context into another agent process through hidden side effects;
  the skill should retrieve context and explicitly include it in the current
  conversation.

## Current Foundation

Already available:

- `mnemonai list --json|--jsonl`
- `mnemonai show <id-or-path> --json`
- Provider-normalized conversation summaries.
- Provider-normalized message details with stable indices, roles, tool call IDs,
  timestamps, model names, project paths, and cwd metadata.
- TUI search scoring in `src/tui/search.rs`.

Missing:

- Headless search over conversations.
- Deterministic message-level excerpt extraction.
- A skill-facing usage guide that teaches agents how to retrieve and cite prior
  context.
- Documentation examples for cross-agent context transfer.

## Design Principles

- Keep `mnemonai` deterministic and model-free.
- Prefer structured JSON as the automation contract.
- Add markdown/plain output only as deterministic formatting over the same data.
- Reuse existing provider loading, caching, filtering, and search logic.
- Preserve source citations in every excerpt: provider, session ID/path, and
  message index range.
- Keep transcript volume bounded by default so agents do not flood their context
  windows.
- Return clear errors for ambiguous session IDs instead of guessing.

## Phase 1: Headless Search

Add:

```text
mnemonai search <query> [--json|--jsonl] [--local|--cwd <path>]
  [--provider <name>] [--since <duration>] [--after <timestamp>]
  [--before <timestamp>] [--limit <n>] [--show-deleted-projects]
```

Behavior:

- Reuse the same scoring behavior as the TUI search where practical.
- Default to JSON array output.
- Apply the same provider, cwd, local/global, deleted-project, and time filters
  as `list`.
- Return conversation-level results first.
- Include all fields from `list` summaries plus a relevance score if the scoring
  code can expose one without distorting the TUI path.

Example:

```bash
mnemonai search "emp tower SP cnc3" --json --limit 20 --since 180d
```

## Phase 2: Deterministic Excerpts

Add:

```text
mnemonai excerpt <id-or-path> [--query <text>] [--from <index>] [--to <index>]
  [--context <n>] [--json|--markdown] [--provider <name>] [--local]
  [--show-deleted-projects] [--match-roles <roles>] [--max-windows <n>]
```

Behavior:

- Resolve the target the same way as `show`.
- With `--from`/`--to`, return that inclusive message range.
- With `--query`, find matching messages in the normalized `messages[]` text
  fields and return bounded windows around each match.
- With `--match-roles`, limit which message roles can trigger query matches
  while still returning neighboring messages for context.
- With `--max-windows`, cap the number of returned windows.
- Merge overlapping windows.
- Include the conversation summary and excerpt windows.
- Preserve original message indices.
- Bound output by default with a conservative context size.

Example:

```bash
mnemonai excerpt "$session_id" --query "emp tower SP" --context 3 --match-roles user,assistant,summary --max-windows 3 --json
```

## Phase 3: Optional Recall Convenience

Consider only after `search` and `excerpt` are exercised by a real skill:

```text
mnemonai recall <query> [search filters...] [--limit-sessions <n>]
  [--context <n>] [--max-chars <n>] [--json|--markdown]
```

`recall` would combine search plus excerpting, but still perform no AI
summarization. The output would be a compact bundle of cited transcript windows
for the agent to reason over.

## Phase 4: Skill

Create a skill such as `mnemonai-memory` after the CLI primitives exist.

The skill should instruct agents to:

- Use `mnemonai search` when the user asks for prior context, past sessions,
  cross-agent continuity, or memory about a topic.
- Prefer `--cwd .` when the current repository is relevant.
- Use global search when the user asks across agents, providers, or topics.
- Start with bounded time windows, then broaden if results are sparse.
- Load excerpts before full transcripts.
- Summarize only the retrieved context relevant to the current task.
- Cite provider, session ID/path, and message ranges.
- Mention when context is incomplete, stale, ambiguous, or not found.
- Avoid copying large raw transcript blocks unless the user asks.

## Phase 5: Documentation

Update `README.md` with:

- Agent/skill retrieval examples.
- `search` output schema.
- `excerpt` output schema.
- A compact workflow for "find context from a previous session and use it in a
  new agent session."

## Test Plan

Add focused unit tests for:

- CLI parsing for `search` and `excerpt`.
- Search filtering and ordering.
- Search JSON and JSONL serialization.
- Excerpt by explicit message range.
- Excerpt by query with context windows.
- Window merge behavior.
- Ambiguous target errors.
- Provider and cwd filters.

Run:

```bash
cargo fmt --all
cargo test --all
cargo clippy -- -W clippy::all
```

## Initial Implementation Order

1. Add shared headless filter structs/helpers so `list` and `search` do not
   duplicate time/cwd/provider filtering.
2. Add `search` command and JSON output.
3. Add `excerpt` command with explicit ranges.
4. Add query-based excerpt windows.
5. Add markdown excerpt formatting if still useful after JSON works.
6. Update README.
7. Create the skill once CLI behavior is stable.
