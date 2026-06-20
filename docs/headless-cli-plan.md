# Headless CLI Plan

## Goal

Add a stable non-interactive CLI surface so tools and Codex skills can analyze
conversation history across Claude Code, Codex, Cursor Agent CLI, and Cursor IDE
without scraping the TUI or terminal render output.

The headless interface should reuse the existing provider, parser, cache, and
search code paths as much as possible. TUI behavior should remain unchanged.

## Current State

`mnemonai` already has partial headless exits:

- `--render <FILE>` renders one JSONL file and exits.
- `--show-dir` prints the Claude project directory for the current directory.
- hidden `--bench-startup` prints provider load timings.

Global and local history browsing still enter the TUI before `--plain`,
`--show-path`, `--show-id`, and `--resume` can produce output. Direct file input
also opens the single-file TUI viewer.

## Proposed CLI

Use subcommands for new machine-readable behavior while keeping existing flags
working for interactive usage.

```text
mnemonai list [--json|--jsonl] [--local] [--provider <name>] [--limit <n>] [--show-deleted-projects]
mnemonai show <id-or-path> [--json|--jsonl|--plain|--ledger] [--show-tools] [--show-thinking]
mnemonai dump [--jsonl] [--local] [--provider <name>] [--since <duration>] [--limit <n>]
mnemonai search <query> [--json|--jsonl] [--local] [--provider <name>] [--limit <n>]
```

Provider names:

- `claude`
- `codex`
- `cursor`
- `cursor-agent`

Default output for new headless subcommands should be JSON or JSONL only. Human
rendering can be explicit on `show`.

## Output Model

### Conversation Summary

Used by `list`, `search`, and `dump` metadata.

```json
{
  "provider": "codex",
  "id": "session-id",
  "path": "/absolute/source/path.jsonl",
  "timestamp": "2026-06-19T10:12:33-07:00",
  "project_name": "mnemonai",
  "project_path": "/Users/bquenin/qube/bquenin/mnemonai",
  "cwd": "/Users/bquenin/qube/bquenin/mnemonai",
  "preview": "short preview",
  "summary": "optional title",
  "model": "optional model",
  "message_count": 12,
  "total_tokens": 123456,
  "duration_minutes": 18,
  "parse_errors": []
}
```

### Conversation Detail

Used by `show` and `dump --include-messages` if added.

```json
{
  "conversation": { "provider": "claude", "id": "session-id" },
  "messages": [
    {
      "role": "user",
      "timestamp": "2026-06-19T10:12:33-07:00",
      "text": "message text",
      "tool_name": null,
      "tool_input": null,
      "tool_result": null,
      "thinking": null
    }
  ]
}
```

Keep the schema intentionally simple for skill consumption. Include raw source
path and provider ID so a caller can re-open or resume with existing tools.

## Implementation Phases

### 1. CLI Shape

- Add a `Command` enum to `src/cli.rs`.
- Preserve current no-subcommand behavior as the interactive default.
- Keep legacy flags available where they already work.
- Add provider parsing as a small enum rather than passing raw strings around.

### 2. Shared Loading Layer

- Extract the global/local loading logic currently embedded in `src/main.rs`
  into reusable functions.
- Return `Vec<Conversation>` without entering the TUI.
- Apply the same sorting, indexing, config merging, local filtering, provider
  filtering, and deleted-project filtering for both TUI and headless commands.
- Keep streaming loading for the TUI path, but allow headless commands to use
  synchronous loading unless benchmark data says this is too slow.

### 3. Serialization Layer

- Add `serde::Serialize` support through dedicated DTO structs instead of
  serializing `Conversation` directly.
- Convert `ProviderKind` to stable lowercase provider keys.
- Serialize paths as strings.
- Keep `parse_errors` available because they are useful for diagnosing broken
  transcripts.

### 4. Message Normalization

- Add a provider-independent message DTO built from `LogEntry`.
- Reuse `Provider::read_entries` for `show`.
- Normalize:
  - text blocks
  - tool calls
  - tool results
  - thinking blocks
  - timestamps
  - assistant model where available
- Avoid exposing renderer-specific ledger spans in JSON.

### 5. Commands

- `list`: load conversations and output summaries.
- `show`: resolve by exact path, exact provider/id pair if later supported, or
  unique ID. Then output detail or explicit human rendering.
- `dump`: output summaries as JSONL first. Add `--include-messages` only if the
  first skill use case needs full transcripts in one pass.
- `search`: reuse `tui::search` scoring so headless search matches TUI ranking.

### 6. Tests

- Unit-test provider enum parsing and JSON DTO serialization.
- Add command-level integration tests using temporary fixture histories.
- Cover:
  - `list --json`
  - `list --jsonl --provider codex`
  - `show <path> --json`
  - `search <term> --json`
  - local filtering
  - deleted-project filtering
  - ambiguous ID errors
- Keep tests independent of real user history.

### 7. Documentation

- Update `README.md` with a short "Headless Usage" section.
- Include examples for `jq` and skill-oriented JSONL processing.
- Document the schema as stable enough for automation.

## Suggested First PR Scope

Keep the first implementation narrow:

1. `mnemonai list --json|--jsonl`
2. `mnemonai show <path-or-id> --json`
3. Provider filtering
4. JSON DTOs and tests

Defer `dump`, `search`, duration parsing like `--since 30d`, and advanced human
render options until the core shape is proven.

## Risks And Tradeoffs

- Refactoring `main.rs` loading code is worth doing, but it should be scoped to
  avoid destabilizing the TUI.
- ID resolution can be ambiguous across providers. The first version should
  return a clear ambiguity error rather than guessing.
- Cursor IDE conversations are database-backed, so `show` should continue using
  the provider abstraction rather than assuming every source is JSONL.
- JSON output should not depend on terminal width, ANSI colors, markdown
  rendering, or pager behavior.
- For very large histories, `dump --include-messages` could be expensive. Start
  with metadata and add full-message streaming deliberately.

## Skill Integration Pattern

A future analysis skill can call:

```bash
mnemonai list --jsonl --limit 500
mnemonai show "$session_id" --json
mnemonai search "failed tests" --json --limit 50
```

The skill should analyze structured fields instead of rendered transcript text,
then report recurring friction such as repeated failed commands, tool errors,
ambiguous instructions, missing project rules, hook failures, stale scripts, or
places where a reusable command or repository rule would reduce repeated work.
