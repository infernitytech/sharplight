#!/usr/bin/env bash
# Downloads the pinned FFmpeg sources, codec libraries and build tools listed under "buildSources" and "buildTools" in
# third_party/sources.json, verifies their SHA-256 and extracts them. They are not vendored in git (FFmpeg alone is ~100 MB of source);
# the pinned archive URL + hash is the record of exactly what was built (and the LGPL source offer points to it).
#   fetch.sh <cache-directory> [name ...]      (default: every build source and tool)
# Prints "<name> <extracted-directory>" per component. An archive whose hash does not match is deleted and the script fails;
# nothing is executed from a download before its hash matched.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cache="${1:?cache directory required}"
shift || true
mkdir -p "$cache/downloads" "$cache/src"

entries="$(python3 -I - "$root/third_party/sources.json" "$@" <<'PY'
import json, sys
with open(sys.argv[1]) as file:
    data = json.load(file)
wanted = set(sys.argv[2:])
for c in data["buildSources"] + data["buildTools"]:
    if wanted and c["name"] not in wanted:
        continue
    print("|".join([c["name"], c["version"], c["url"], c["sha256"], c["archiveRoot"]]))
PY
)"

while IFS='|' read -r name version url sha256 archive_root; do
    [ -n "$name" ] || continue
    file="$cache/downloads/$name-$version-$(basename "$url")"
    destination="$cache/src/$name-$version"
    if [ ! -f "$destination/.sharplight-verified" ]; then
        if [ ! -f "$file" ]; then
            curl -fsSL --retry 3 -o "$file.part" "$url"
            mv "$file.part" "$file"
        fi

        actual="$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$file" | cut -d' ' -f1)"
        if [ "$actual" != "$sha256" ]; then
            rm -f "$file"
            echo "SHA-256 mismatch for $name $version: expected $sha256, got $actual" >&2
            exit 1
        fi

        rm -rf "$destination" "$destination.extract"
        mkdir -p "$destination.extract"
        tar -xf "$file" --no-same-owner -C "$destination.extract"
        mv "$destination.extract/$archive_root" "$destination"
        rm -rf "$destination.extract"
        touch "$destination/.sharplight-verified"
    fi

    echo "$name $destination"
done <<< "$entries"
