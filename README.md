# mnemonai

Universal AI coding conversation history browser. Search, browse, and resume conversations across multiple AI coding tools from a single TUI.

## Supported Tools

| Tool | History Format | Resume Support |
|------|---------------|----------------|
| **Claude Code** | JSONL files in `~/.claude/projects/` | `claude --resume <session-id>` |
| **Cursor** | SQLite in workspace storage | Bridge extension + `cursor://` URI |

## Supported Platforms

- macOS (Apple Silicon and Intel)
- Linux (x86_64)

## Install

### Homebrew

```bash
brew install bquenin/mnemonai/mnemonai
```

### Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/bquenin/mnemonai/main/scripts/install.sh | bash
```

## Usage

```bash
# Launch the TUI (shows all conversations across all providers)
mnemonai

# Only show conversations from the current project directory
mnemonai --local
```

## Keyboard Shortcuts

### List View

| Key | Action |
|-----|--------|
| Type | Fuzzy search conversations |
| `Up/Down` or `j/k` | Navigate list |
| `Enter` | View conversation |
| `r` | Resume conversation in original tool |
| `Tab` | Cycle provider filter (All → Claude → Cursor) |
| `Esc` | Clear search / Quit |
| `q` | Quit |

### Detail View

| Key | Action |
|-----|--------|
| `Up/Down` or `j/k` | Scroll |
| `Page Up/Down` | Scroll fast |
| `g/G` | Jump to top/bottom |
| `r` | Resume conversation |
| `Esc` or `q` | Back to list |

## Configuration

Create `~/.config/mnemonai/config.toml`:

```toml
[display]
show_tools = false      # Show tool-use messages
relative_time = true    # "2 hours ago" vs "2026-02-18 14:30"

[providers.claude]
enabled = true

[providers.cursor]
enabled = true
```

## Releasing

1. Bump the version in `Cargo.toml`
2. Commit and push
3. Tag and push the tag:

```bash
git tag v0.x.x
git push origin v0.x.x
```

The GitHub Actions release workflow triggers on `v*` tags, builds platform binaries, creates the GitHub release, and updates the Homebrew tap.

## Acknowledgments

mnemonai is a fork of [claude-history](https://github.com/raine/claude-history) by [Raine Virta](https://github.com/raine).

## License

MIT
