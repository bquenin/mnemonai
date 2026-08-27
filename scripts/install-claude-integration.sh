#!/usr/bin/env bash
# Install the mnemonai Claude Code integration: the `handoff` skill, the
# session-digest script, and the context-size warning hook. Everything is symlinked
# from this checkout, so `git pull` updates the installed copies. Re-run this
# script from the canonical checkout if the repo moves (e.g. after developing
# in a worktree).
#
# Idempotent. Installs:
#   ~/.claude/skills/handoff                     -> skills/handoff
#   ~/.claude/scripts/session-digest.sh          -> scripts/session-digest.sh
#   ~/.claude/scripts/mnemonai-context-warn.sh   -> hooks/context-warn.sh
#
# The hook must also be wired in ~/.claude/settings.json (printed at the end;
# not modified automatically).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
claude_dir="${HOME}/.claude"

link() { # target link_path
  local target="$1" link_path="$2"
  if [ -e "$link_path" ] && [ ! -L "$link_path" ]; then
    echo "replacing non-symlink $link_path (backup: ${link_path}.bak)"
    rm -rf "${link_path}.bak"
    mv "$link_path" "${link_path}.bak"
  fi
  ln -sfn "$target" "$link_path"
  echo "  $link_path -> $target"
}

mkdir -p "$claude_dir/skills" "$claude_dir/scripts" "$HOME/.claude/handoffs"

echo "Installing mnemonai Claude Code integration from $repo_root:"
link "$repo_root/skills/handoff"          "$claude_dir/skills/handoff"
link "$repo_root/scripts/session-digest.sh"  "$claude_dir/scripts/session-digest.sh"
link "$repo_root/hooks/context-warn.sh"      "$claude_dir/scripts/mnemonai-context-warn.sh"

command -v mnemonai >/dev/null 2>&1 || \
  echo "WARNING: mnemonai binary not found on PATH — session-digest.sh needs it."

cat <<'EOF'

Hook wiring for ~/.claude/settings.json (merge into any existing "hooks"):

  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "~/.claude/scripts/mnemonai-context-warn.sh" } ] }
    ]
  }

Threshold override (default 200000 tokens): set MNEMONAI_CTX_WARN_TOKENS in
settings.json "env".
EOF
