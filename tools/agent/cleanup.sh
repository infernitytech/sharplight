#!/usr/bin/env bash
# Fast periodic cleanup (no du over a network volume): prune stale worktrees and temp files older than 6 hours; report free space.
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
[ -d "$SL_REPO/.git" ] && git -C "$SL_REPO" worktree prune
find "$SL_WORK/tmp" -mindepth 1 -maxdepth 1 -mmin +360 -exec rm -rf {} + 2>/dev/null
find "$TMPDIR" -mindepth 1 -maxdepth 1 -mmin +360 -exec rm -rf {} + 2>/dev/null
df -h "$SL_LOCAL" "$SL_DATA" 2>/dev/null | tail -n +2
free -g | sed -n 2p
