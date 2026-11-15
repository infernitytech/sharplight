#!/usr/bin/env bash
# Copies the Claude Code memory notes in docs/handoff/memory into this clone's Claude Code project memory directory
# (~/.claude/projects/<clone path with / replaced by ->/memory). Existing notes with the same name are overwritten.
set -euo pipefail
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
key="$(printf '%s' "$root" | sed 's#[^a-zA-Z0-9]#-#g')"
dest="${HOME}/.claude/projects/${key}/memory"
mkdir -p "$dest"
cp "$root"/docs/handoff/memory/*.md "$dest"/
echo "Restored $(ls "$root"/docs/handoff/memory/*.md | wc -l) memory notes to $dest"
