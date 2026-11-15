#!/usr/bin/env bash
# Writes Markdown release notes (the commit subjects since the previous release tag, or the last 30 commits) for Velopack.
# Usage: packaging/release-notes.sh <output file>
set -euo pipefail

out="${1:?output file required}"
previous="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
{
    echo "## Changes"
    echo
    if [ -n "$previous" ]; then
        git log --no-merges --pretty='- %s' "$previous..HEAD"
    else
        git log --no-merges --pretty='- %s' -n 30
    fi
} > "$out"
if [ "$(wc -l < "$out")" -le 2 ]; then
    echo "- Maintenance release." >> "$out"
fi
