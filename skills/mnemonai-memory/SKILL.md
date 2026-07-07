---
name: mnemonai-memory
description: Retrieve cited context from prior AI coding agent sessions with the mnemonai CLI. Use when a user asks for memory, past-session context, cross-agent continuity, what happened before on a topic, resuming work with context from another agent/session, or finding previous Claude/Codex/Cursor/Cursor Agent conversations relevant to the current task.
---

# Mnemonai Memory

Use `mnemonai` as a deterministic retrieval backend. `mnemonai` finds prior
sessions and transcript excerpts; the current agent does all interpretation,
summarization, and task application.

## Workflow

1. Resolve the binary once.

```bash
MNEMONAI_BIN="${MNEMONAI_BIN:-mnemonai}"
"$MNEMONAI_BIN" search mnemonai --json --limit 0 >/dev/null
"$MNEMONAI_BIN" help excerpt | grep -q 'Usage: .* excerpt'
```

Do not use `--version` alone as the capability check; local development builds
and installed releases can have the same version string. If either probe fails
and the current repository has `target/debug/mnemonai`, retry with:

```bash
MNEMONAI_BIN="$PWD/target/debug/mnemonai"
"$MNEMONAI_BIN" search mnemonai --json --limit 0 >/dev/null
"$MNEMONAI_BIN" help excerpt | grep -q 'Usage: .* excerpt'
```

If the commands are still unavailable, fall back to `list`/`show` only when the
task is still practical. Otherwise report that this skill needs a `mnemonai`
build with headless `search` and `excerpt` support.

2. Choose scope from the user's request.

- Current repository context: prefer `--cwd .`.
- Cross-agent or broad memory: omit `--cwd`.
- Specific provider: add `--provider claude|codex|cursor|cursor-agent`.
- Recent work: start with `--since 30d`; broaden if results are sparse.
- Explicit old topic: use a wider window such as `--since 180d` or omit time.

3. Search for candidate sessions.

```bash
"$MNEMONAI_BIN" search "topic words" --json --cwd . --limit 10
```

Inspect provider, id, path, timestamp, project/cwd, summary, preview, and score.
Scores are comparable only within the same search result set.

4. Load excerpts, not full transcripts, first.

```bash
"$MNEMONAI_BIN" excerpt "$session_id" --query "topic words" --context 3 --match-roles user,assistant,summary --max-windows 3 --json --provider codex
```

For memory retrieval, prefer `--match-roles user,assistant,summary` so noisy
tool output does not trigger windows by itself. Increase or remove
`--max-windows` only when the first windows are clearly incomplete. Pass
`--provider` when IDs may be ambiguous. Use explicit ranges when a prior result
or user request gives message indices:

```bash
"$MNEMONAI_BIN" excerpt "$session_id" --from 24 --to 39 --json --provider codex
```

Use `show <id-or-path> --json` only when excerpts are insufficient.

5. Summarize retrieved context for the current task.

Include only relevant information:

- Facts or decisions from prior sessions.
- Files, repositories, commands, services, tickets, or PRs mentioned.
- Failed approaches and successful recoveries.
- Open questions or unresolved work.
- Differences between old context and the current request.

Always cite sources with provider, session ID or path, and message range, for
example `codex:019ee... messages 24-39`.

## Output Style

Keep the response compact. Do not paste large raw transcript blocks unless the
user asks. State when no matching sessions were found, when matches are weak, or
when context may be stale.

Preferred shape:

```markdown
Prior context found:
- ...

Relevant decisions:
- ...

Useful references:
- ...

Sources:
- codex:<session-id> messages 24-39
- claude:<session-id> messages 8-15
```
