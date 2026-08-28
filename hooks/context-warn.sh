#!/usr/bin/env bash
# Claude Code UserPromptSubmit hook: warn when the session's context is
# getting expensive to carry.
#
# The context-window meter answers "am I about to run out of room"; on a 1M
# window a 400K context renders as a comfortable 40% while costing ~$0.40 to
# send every further message. This hook watches the number that actually
# escalates — context size in tokens — and shows a one-line warning when the
# session crosses 200K, then again at each further 100K band. One warning per
# band per session (state in ~/.claude/cache/context-warn/).
#
# Wire in ~/.claude/settings.json:
#   "hooks": { "UserPromptSubmit": [ { "hooks": [
#     { "type": "command", "command": "~/.claude/scripts/mnemonai-context-warn.sh" }
#   ] } ] }

set -u

input_json="$(cat)"

WARN_AT_TOKENS="${MNEMONAI_CTX_WARN_TOKENS:-200000}"   # first warning threshold
BAND_TOKENS=100000                                     # re-warn every N further

session_id="$(printf '%s' "$input_json" | jq -r '.session_id // empty' 2>/dev/null)"
transcript_path="$(printf '%s' "$input_json" | jq -r '.transcript_path // empty' 2>/dev/null)"
[ -n "$transcript_path" ] && [ -f "$transcript_path" ] || exit 0

# Context of the NEXT call = last assistant turn's total input tokens.
read -r ctx model < <(
  tac "$transcript_path" 2>/dev/null \
    | jq -rc 'select(.type=="assistant" and .message.usage != null)
              | [((.message.usage.input_tokens // 0)
                  + (.message.usage.cache_creation_input_tokens // 0)
                  + (.message.usage.cache_read_input_tokens // 0)),
                 (.message.model // "?")] | @tsv' 2>/dev/null \
    | head -n 1 | tr '\t' ' '
)
[ -n "${ctx:-}" ] && [ "$ctx" -ge "$WARN_AT_TOKENS" ] 2>/dev/null || exit 0

band=$(( ctx / BAND_TOKENS ))
state_dir="${HOME}/.claude/cache/context-warn"
state_file="${state_dir}/${session_id:-unknown}"
last_band=0
[ -r "$state_file" ] && read -r last_band < "$state_file" 2>/dev/null
[ "$band" -gt "${last_band:-0}" ] 2>/dev/null || exit 0
mkdir -p "$state_dir" 2>/dev/null && printf '%s\n' "$band" > "$state_file"

# $/turn floor: whole context re-read at the cache-read rate (0.1x input).
case "${model:-}" in
  *fable*|*mythos*) rate=10 ;;
  *opus*)           rate=5  ;;
  *sonnet-4-6*)     rate=3  ;;
  *sonnet*)         rate=2  ;;
  *haiku*)          rate=1  ;;
  *)                rate=0  ;;
esac
cost=""
if [ "$rate" -gt 0 ]; then
  cost="$(awk -v t="$ctx" -v r="$rate" 'BEGIN{printf " (~$%.2f/turn on this model)", t*r*0.1/1000000}')"
fi

ctx_disp="$(awk -v t="$ctx" 'BEGIN{printf "%dK", t/1000}')"
sid_short="${session_id:0:8}"

jq -cn --arg msg "context is ${ctx_disp}${cost}. Options: say \"handoff\" to checkpoint, /compact to shrink in place, or start fresh later with \"resume session ${sid_short}\"." \
  '{systemMessage: $msg, suppressOutput: true}'
exit 0
