#!/usr/bin/env bash
# Re-fetches the pinned third-party sources listed in sources.json, verifies their SHA-256 and replaces the vendored copies.
#   update.sh [component ...]     (default: all components)
# To change a version: edit url, version, archiveRoot and sha256 in sources.json (and the table in README.md), then run this.
# Downloads are untrusted until the checksum matches; each run uses its own fresh directory under $SLA_DOWNLOADS
# (default: a new mktemp directory) and nothing is executed from it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
downloads="${SLA_DOWNLOADS:-}"
if [ -z "$downloads" ]; then
    downloads="$(mktemp -d)"
else
    downloads="$downloads/run-$(date +%s)"
    mkdir -p "$downloads"
fi

# component fields: name|url|sha256|archiveRoot|copy (space separated)|prune (space separated)
components="$(python3 -I - "$here/sources.json" "$@" <<'PY'
import json, sys
with open(sys.argv[1]) as file:
    data = json.load(file)
wanted = set(sys.argv[2:])
for c in data["components"]:
    if wanted and c["name"] not in wanted:
        continue
    print("|".join([c["name"], c["url"], c["sha256"], c["archiveRoot"], " ".join(c["copy"]), " ".join(c.get("prune", []))]))
PY
)"

while IFS='|' read -r name url sha256 root copy prune; do
    [ -n "$name" ] || continue
    echo "== $name"
    work="$downloads/$name"
    mkdir -p "$work/extract"
    curl -fsSL --retry 3 -o "$work/archive.tar.gz" "$url"
    actual="$(sha256sum "$work/archive.tar.gz" | cut -d' ' -f1)"
    if [ "$actual" != "$sha256" ]; then
        echo "SHA-256 mismatch for $name: expected $sha256, got $actual" >&2
        exit 1
    fi

    tar -xzf "$work/archive.tar.gz" --no-same-owner -C "$work/extract"
    destination="$here/$name"
    rm -rf "$destination"
    mkdir -p "$destination"
    for item in $copy; do
        mkdir -p "$destination/$(dirname "$item")"
        cp -r "$work/extract/$root/$item" "$destination/$item"
    done

    for pattern in $prune; do
        find "$destination" -name "$pattern" -prune -exec rm -rf {} +
    done
done <<< "$components"
echo "done; review the diff and update README.md if versions changed"
