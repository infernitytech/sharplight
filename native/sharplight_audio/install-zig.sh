#!/usr/bin/env bash
# Installs the pinned zig toolchain (used only as a cross compiler) for an x86_64 Linux host.
#   install-zig.sh <install-directory>     prints the path of the zig binary on the last line
set -euo pipefail

version="0.15.2"
sha256="02aa270f183da276e5b5920b1dac44a63f1a49e55050ebde3aecc9eb82f93239"
url="https://ziglang.org/download/$version/zig-x86_64-linux-$version.tar.xz"
destination="${1:?install directory required}"

if [ ! -x "$destination/zig-x86_64-linux-$version/zig" ]; then
    mkdir -p "$destination"
    download="$(mktemp -d)"
    curl -fsSL --retry 3 -o "$download/zig.tar.xz" "$url"
    actual="$(sha256sum "$download/zig.tar.xz" | cut -d' ' -f1)"
    if [ "$actual" != "$sha256" ]; then
        echo "SHA-256 mismatch for zig $version: expected $sha256, got $actual" >&2
        exit 1
    fi

    tar -xJf "$download/zig.tar.xz" --no-same-owner -C "$destination"
    rm -rf "$download"
fi

echo "$destination/zig-x86_64-linux-$version/zig"
