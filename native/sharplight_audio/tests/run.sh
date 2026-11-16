#!/usr/bin/env bash
# Builds and runs the native tests against an already built library.
#   run.sh <library-directory> <fixtures-directory> [fuzz-iterations]
# Uses zig cc (ZIG, default zig). For the sanitizer variant build the library with `build.sh host-sanitize` and pass out/linux-x64-asan.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_dir="$1"
fixtures="$2"
iterations="${3:-200}"
zig="${ZIG:-zig}"
flags=(-std=gnu11 -O1 -g -Wall -Wextra -Werror -I"$here/../include")
case "$lib_dir" in
    *-asan) flags+=(-fsanitize=address,undefined) ;;
esac
mkdir -p "$lib_dir/../obj"
exe="$lib_dir/../obj/sla_test-$(basename "$lib_dir")"
nice -n 15 "$zig" cc "${flags[@]}" "$here/test_main.c" -L"$lib_dir" -lsharplight_audio -lm "-Wl,-rpath,$lib_dir" -o "$exe"
exec "$exe" "$fixtures" "$iterations"
