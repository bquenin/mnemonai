#!/usr/bin/env bash
# session-digest.sh <session-id-or-prefix-or-path> [max-chars]
#
# Distill a past agent session into a compact conversation digest for resuming
# the work in a FRESH session: user prompts + assistant prose, no tool calls,
# no tool results, no thinking. Recent messages are kept fuller than old ones,
# and the total is capped (default 120000 chars ~ 30K tokens) so reading the
# digest costs cents instead of the dollars of resuming a marathon session.
#
# Provider-agnostic: anything `mnemonai show` can resolve (Claude Code, Codex,
# Cursor, Cursor Agent). Short id prefixes are resolved via `mnemonai list`.
# Used by the `handoff` skill's "resume session <id>" flow.

set -euo pipefail

ref="${1:?usage: session-digest.sh <session-id-or-prefix-or-path> [max-chars]}"
max_chars="${2:-120000}"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if ! mnemonai show "$ref" --json > "$tmp" 2>/dev/null; then
  # Not a full id or path — try resolving it as an id prefix.
  full_id="$(mnemonai list --json --since 365d --limit 2000 2>/dev/null \
    | python3 -c '
import json, sys
prefix = sys.argv[1]
ids = sorted({r["id"] for r in json.load(sys.stdin) if str(r.get("id", "")).startswith(prefix)})
if len(ids) == 1:
    print(ids[0])
elif len(ids) > 1:
    print(f"ambiguous prefix {prefix!r}: " + ", ".join(ids[:5]), file=sys.stderr)
' "$ref")"
  [ -n "$full_id" ] || { echo "session-digest: no session found for '$ref'" >&2; exit 1; }
  mnemonai show "$full_id" --json > "$tmp"
fi

python3 - "$tmp" "$max_chars" <<'PYEOF'
import json, sys

path, max_chars = sys.argv[1], int(sys.argv[2])
d = json.load(open(path))
conv = d.get("conversation") or {}
def keep(m):
    if m.get("role") not in ("user", "assistant"):
        return False
    t = (m.get("text") or "").strip()
    # Drop harness noise recorded as user messages: task notifications,
    # command echoes, system reminders, injected skill bodies.
    return bool(t) and not t.startswith("<") and "Base directory for this skill" not in t

msgs = [m for m in d.get("messages") or [] if keep(m)]

# Budget: the last RECENT messages verbatim, older ones clipped; drop from the
# head if still over budget. The tail is where the resumable state lives.
RECENT, OLD_CLIP = 40, 300
n = len(msgs)
out = []
for i, m in enumerate(msgs):
    text = m["text"].strip()
    if i < n - RECENT and len(text) > OLD_CLIP:
        text = text[:OLD_CLIP] + " [...]"
    who = "USER" if m["role"] == "user" else "ASSISTANT"
    ts = str(m.get("timestamp") or "")[:16]
    out.append(f"[{who} {ts}]\n{text}\n")

while out and sum(len(x) for x in out) > max_chars:
    out.pop(0)

sid = conv.get("id", "?")
print(f"# Session digest: {sid} ({conv.get('provider', '?')})")
print(f"# cwd: {conv.get('cwd')}  project: {conv.get('project_path')}")
print(f"# messages: {n} user+assistant (showing {len(out)}); tool output excluded.")
print(f"# For a specific detail: mnemonai show {sid} --grep <term> --context 2 --json")
print()
print("\n".join(out))
PYEOF
