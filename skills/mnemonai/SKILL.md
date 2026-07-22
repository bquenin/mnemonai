---
name: mnemonai
description: Search past AI agent sessions (Claude Code, Codex, Cursor IDE, Cursor Agent) and Slack history for prior discussions, decisions, and solutions. Use when the user asks "have we seen/discussed/solved this before?", wants prior context or a past recommendation pulled into this session, or references earlier work they cannot locate.
---

# mnemonai: retrieve prior context from agent sessions and Slack

## When to use
- "Do you remember…", "have we already…", "we solved this before…", "find the thread/session where…"
- Before proposing a fix for an issue that smells recurring.

## Tools
- `mnemonai` (installed): headless content search over ALL local agent sessions.
- Slack tooling: workspace search + thread reading.

## Method
1. Build TWO keyword sets before searching:
   - **Concrete**: artifact names from the immediate context — repos, endpoints, error strings, ticket ids.
   - **Abstract**: the problem-class vocabulary — ask "what is this an instance of?" (e.g.
     "reusable workflow secrets OIDC", not just the endpoint name). Prior discussions rarely
     reuse today's nouns.
2. Search sessions with EACH set:
   `mnemonai search <words> --since 365d --limit 5 --json`
   - Every word is a case-insensitive substring match (infix: `flow` matches `workflow`);
     multi-word queries AND together. Prefer distinctive terms; short generic words over-match.
   - Zero hits: vary terms (synonyms, component names) before giving up. Scope with
     --provider/--cwd only when confident.
   - Exclude the live session when its id is known: `--exclude-session <id>`; otherwise ignore
     a self-hit (a result timestamped now, matching what you just wrote).
3. Drill into shortlisted sessions WITHOUT dumping transcripts:
   `mnemonai show <id> --grep <term> --context 2 --json`
4. Slack, with the SAME two vocabularies — and mine the session hits first: sessions often
   contain the Slack permalinks you're looking for. Read candidate threads fully before citing.
5. Synthesize: the prior problem, what was decided/recommended, by whom, and what may have
   changed since. CITE everything: session id + date + provider, Slack permalinks, ticket ids.
   Offer `mnemonai show <id> --json` for a deep dive.

## Pitfalls
- `list` previews do NOT search content — use `search`.
- Never `show` without `--grep` on large sessions (hundreds of messages).
- A recommendation found in one session may have been superseded — check for later
  sessions/threads on the same terms before presenting it as current.
