#!/usr/bin/env bash
# Claude Code SessionEnd hook: when a session ends carrying a large context,
# print how to resume it CHEAPLY — via the mnemonai-backed "resume session"
# flow (a fresh session reads a compact digest of this one) — instead of
# `claude -r <id>`, which reloads the full context and its full per-turn cost.
#
# SessionEnd stdout is not shown in the terminal, so this writes to /dev/tty
# (falls back to notify-send when no tty is available).
#
# Wire in ~/.claude/settings.json:
#   "hooks": { "SessionEnd": [ { "hooks": [
#     { "type": "command", "command": "~/.claude/scripts/mnemonai-session-end.sh" }
#   ] } ] }

set -u

input_json="$(cat)"

NOTIFY_AT_TOKENS="${MNEMONAI_CTX_WARN_TOKENS:-200000}"

session_id="$(printf '%s' "$input_json" | jq -r '.session_id // empty' 2>/dev/null)"
transcript_path="$(printf '%s' "$input_json" | jq -r '.transcript_path // empty' 2>/dev/null)"
[ -n "$transcript_path" ] && [ -f "$transcript_path" ] || exit 0

ctx="$(
  tac "$transcript_path" 2>/dev/null \
    | jq -rc 'select(.type=="assistant" and .message.usage != null)
              | ((.message.usage.input_tokens // 0)
                 + (.message.usage.cache_creation_input_tokens // 0)
                 + (.message.usage.cache_read_input_tokens // 0))' 2>/dev/null \
    | head -n 1
)"
[ -n "${ctx:-}" ] && [ "$ctx" -ge "$NOTIFY_AT_TOKENS" ] 2>/dev/null || exit 0

ctx_disp="$(awk -v t="$ctx" 'BEGIN{printf "%dK", t/1000}')"
sid_short="${session_id:0:8}"

msg="session ${sid_short} ended with ${ctx_disp} of context.
  cheap resume:  claude  ->  \"resume session ${sid_short}\"   (fresh session + mnemonai digest)
  full resume:   claude -r ${session_id}   (reloads all ${ctx_disp}, full cost per turn)"

if ! { printf '\n%s\n' "$msg" > /dev/tty; } 2>/dev/null; then
  command -v notify-send >/dev/null 2>&1 && notify-send "Claude Code" "$msg" 2>/dev/null || :
fi
exit 0
