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

Press `?` at any time to open the help overlay.

### List View

| Key | Action |
|-----|--------|
| Type | Fuzzy search conversations |
| `Up/Down` | Navigate list |
| `Home/End` | Jump to first/last |
| `Page Up/Down` | Page navigation |
| `Ctrl+D/U` | Half page down/up |
| `Ctrl+N/P` | Next/prev (emacs-style) |
| `Enter` | View conversation |
| `Ctrl+R` | Resume conversation in original tool |
| `Ctrl+X` | Delete conversation |
| `Ctrl+O` | Select and exit |
| `Ctrl+W` | Delete word |
| `Esc` | Quit |

### Detail View

| Key | Action |
|-----|--------|
| `Up/Down` or `j/k` | Scroll |
| `d/u` or `Ctrl+D/U` | Half page down/up |
| `Page Up/Down` | Page navigation |
| `g/G` or `Home/End` | Jump to top/bottom |
| `/` | Search within conversation |
| `n/N` | Next/prev search match |
| `t` | Toggle tool display |
| `T` | Toggle thinking blocks |
| `i` | Toggle timestamps and durations |
| `p` | Show file path |
| `Y` | Copy path to clipboard |
| `I` | Copy session ID to clipboard |
| `e` | Export to file |
| `y` | Copy to clipboard |
| `Ctrl+R` | Resume conversation |
| `Ctrl+X` | Delete conversation |
| `Esc` or `q` | Back to list |

## Configuration

Create `~/.config/mnemonai/config.toml`:

```toml
# Only show conversations from the current project directory
local = false

# Hide projects whose name contains any of these strings
exclude = ["some-project", "another-project"]

[display]
no_tools = false              # Hide tool-use messages
relative_time = true          # "2 hours ago" vs "2026-02-18 14:30"
last = false                  # Show last messages in preview (vs first)
show_thinking = false         # Show thinking blocks
plain = false                 # Plain text output without formatting
pager = false                 # Use pager (less) for output
show_deleted_projects = false # Include conversations from deleted directories

[resume]
default_args = []             # Default args passed to 'claude --resume'
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
