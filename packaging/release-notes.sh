#!/usr/bin/env bash
# Writes Markdown release notes for Velopack: the curated docs/release-notes/v<version>.md of a stable version, otherwise the
# commit subjects since the previous release tag (or the last 30 commits).
# Usage: packaging/release-notes.sh <output file>
set -euo pipefail

out="${1:?output file required}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
# A stable version (no prerelease suffix in version.json) with curated notes in docs/release-notes/v<version>.md uses those.
version="$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' "$repo/version.json" | head -n 1)"
curated="$repo/docs/release-notes/v$version.md"
if [[ -n "$version" && "$version" != *-* && -f "$curated" ]]; then
    cp "$curated" "$out"
    exit 0
fi
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
