#!/usr/bin/env bash
# Claude Code SessionEnd hook: when a session ends carrying a large context,
# print how to resume it CHEAPLY — via the mnemonai-backed "resume session"
# flow (a fresh session reads a compact digest of this one) — instead of
# `claude -r <id>`, which reloads the full context and its full per-turn cost.
#
# SessionEnd stdout is not shown in the terminal, and in fullscreen (alt
# screen) mode anything written to /dev/tty during teardown is wiped when the
# terminal restores the main screen. So the tty write is detached (setsid) and
# delayed ~1s to land AFTER Claude Code has exited. Every firing is also
# appended to ~/.claude/cache/session-end.log, which doubles as the fallback
# when no tty is available and as the trace for debugging.
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
reason="$(printf '%s' "$input_json" | jq -r '.reason // "?"' 2>/dev/null)"

log_file="${HOME}/.claude/cache/session-end.log"
mkdir -p "${HOME}/.claude/cache" 2>/dev/null
trace() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log_file" 2>/dev/null || :; }

[ -n "$transcript_path" ] && [ -f "$transcript_path" ] \
  || { trace "fired session=${session_id:-?} reason=${reason} -> no transcript, skip"; exit 0; }

ctx="$(
  tac "$transcript_path" 2>/dev/null \
    | jq -rc 'select(.type=="assistant" and .message.usage != null)
              | ((.message.usage.input_tokens // 0)
                 + (.message.usage.cache_creation_input_tokens // 0)
                 + (.message.usage.cache_read_input_tokens // 0))' 2>/dev/null \
    | head -n 1
)"
if ! { [ -n "${ctx:-}" ] && [ "$ctx" -ge "$NOTIFY_AT_TOKENS" ] 2>/dev/null; }; then
  trace "fired session=${session_id:-?} reason=${reason} ctx=${ctx:-?} -> below threshold, silent"
  exit 0
fi

ctx_disp="$(awk -v t="$ctx" 'BEGIN{printf "%dK", t/1000}')"
sid_short="${session_id:0:8}"

msg="session ${sid_short} ended with ${ctx_disp} of context.
  cheap resume:  claude  ->  \"resume session ${sid_short}\"   (fresh session + mnemonai digest)
  full resume:   claude -r ${session_id}   (reloads all ${ctx_disp}, full cost per turn)"

trace "fired session=${session_id:-?} reason=${reason} ctx=${ctx} -> notified"
printf '%s\n\n' "$msg" >> "$log_file" 2>/dev/null || :

# Print AFTER Claude Code has exited: detach from the hook process (setsid,
# closed stdio) and delay so the alt-screen teardown has restored the main
# screen. Otherwise the message lands on the alt screen and is wiped.
# setsid drops the controlling terminal, so /dev/tty stops resolving in the
# child — capture the concrete device (/dev/pts/N) first; the controlling
# terminal is a process attribute, readable even though our stdio are pipes.
tty_name="$(ps -o tty= -p $$ 2>/dev/null | tr -d ' ')"
tty_path=""
[ -n "$tty_name" ] && [ "$tty_name" != "?" ] && [ -w "/dev/$tty_name" ] && tty_path="/dev/$tty_name"

delay="${MNEMONAI_SESSION_END_DELAY:-1}"
setsid bash -c '
  sleep "$1"
  { [ -n "$3" ] && printf "\n%s\n" "$2" > "$3"; } 2>/dev/null ||
    { command -v notify-send >/dev/null 2>&1 && notify-send "Claude Code" "$2"; } 2>/dev/null || :
' _ "$delay" "$msg" "$tty_path" >/dev/null 2>&1 < /dev/null &
exit 0
