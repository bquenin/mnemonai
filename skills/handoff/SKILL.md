---
name: handoff
description: Cheap session continuity via mnemonai. Use when the user says "resume session <id>", "resume", "pick up where we left off", or "continue the X work" in a NEW session; or says "handoff", "checkpoint", "save state", or is ending a work day mid-plan. Distills a past session's transcript (or the live session) into a small handoff file so fresh sessions start from ~2K tokens instead of resuming a marathon session's full context.
---

# handoff: cheap session continuity

Working state lives in `~/.claude/handoffs/<slug>.md` — one file per ongoing
workstream, overwritten at each checkpoint. The transcript on disk is the
archive; mnemonai can search it. Handoffs need no git history.

## Resuming a past session ("resume session <id>" or "resume the X work")

The past session does NOT need to have written anything — the transcript is
the source of truth and you distill it post-hoc:

1. Identify the session. An id (or 8-char prefix) resolves directly. A
   workstream name: check `ls -t ~/.claude/handoffs/` for an existing handoff
   first, else `mnemonai search <terms> --since 30d --limit 5 --json`.
2. If a fresh handoff for this workstream already exists and is newer than
   the session's last activity, just read it and skip to step 5.
3. Run `~/.claude/scripts/session-digest.sh <id>` and read its output — user
   prompts + assistant prose only, capped at ~30K tokens, no tool output.
4. Write/overwrite `~/.claude/handoffs/<slug>.md` from the digest (template
   below), so the NEXT resume skips the digest entirely.
5. Read only the files the first next-step actually needs — not everything
   preemptively. For any missing detail, drill into the source with
   `mnemonai show <id> --grep <term> --context 2 --json` — never dump the
   full transcript.
6. Confirm the next step with the user in one line, then proceed.

## Writing a handoff (end of session / before a long break)

Write `~/.claude/handoffs/<slug>.md` where `<slug>` names the workstream
(e.g. `runner-consolidation`), not the date. Overwrite any existing file for
the same workstream. Keep the whole file under ~150 lines. Structure:

```markdown
# <workstream title>
Updated: <ISO date> | Session: <current session id if known>

## Goal
One paragraph. Include ticket IDs, PR links, Slack permalinks.

## State
Where execution stands RIGHT NOW. Which plan steps are done / in flight /
blocked. Exact branch names, worktree paths, cluster/app names.

## Decisions
Bullet list of decisions made and REJECTED alternatives with the why
(rejections are what you re-litigate if you lose them).

## Next steps
Ordered. First entry = the very next action, specific enough to execute cold.

## Key files & commands
Paths read/edited that matter; commands that worked (exact invocations).

## Recall hints
2-4 mnemonai search term sets for details deliberately left out, e.g.:
`mnemonai search "egress gateway TLS" --since 30d --limit 5 --json`
```

Content rules:
- Decisions and rejections are the highest-value content; tool output is the
  lowest — never paste tool output, link or name the source instead.
- Write for a reader with ZERO context: the next session starts cold.
- After writing, tell the user the file path and that it's safe to end the
  session.

## Hygiene

When a workstream is finished (PR merged, ticket closed), delete its handoff
file. If a handoff is >30 days old on resume, verify branches/tickets still
exist before acting on it.
