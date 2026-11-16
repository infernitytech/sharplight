#!/usr/bin/env bash
# Cross-compiles the Sharplight video library (libvpx VP8/VP9 encoders, generic C) with zig cc.
#   build.sh <rid|all|host-sanitize> [--jobs N]
# RIDs: linux-x64 linux-arm64 win-x64 win-arm64 osx-arm64 (all cross-compiled with zig on any host; libvpx needs no platform SDK).
# "host-sanitize" builds linux-x64 with ASan+UBSan for the C test.
# Environment: ZIG (path to zig, default: zig on PATH), SLV_OUT (output root, default native/sharplight_video/out),
#              SLV_BUILD (object cache, default $SLV_OUT/obj), SLV_FORCE (rebuild everything).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
tp="$root/third_party"
zig="${ZIG:-zig}"
out_root="${SLV_OUT:-$here/out}"
build_root="${SLV_BUILD:-$out_root/obj}"
jobs=4
target_arg="${1:-all}"
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs) jobs="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$build_root/zig-global}"
export ZIG_LOCAL_CACHE_DIR="${ZIG_LOCAL_CACHE_DIR:-$build_root/zig-local}"

compile_target() {
    local rid="$1"
    local sanitize="${2:-0}"
    local triple lib link=() extra=()
    case "$rid" in
        linux-x64) triple="x86_64-linux-gnu.2.28"; lib="libsharplight_video.so"; link=(-lpthread -lm) ;;
        linux-arm64) triple="aarch64-linux-gnu.2.28"; lib="libsharplight_video.so"; link=(-lpthread -lm) ;;
        win-x64) triple="x86_64-windows-gnu"; lib="sharplight_video.dll" ;;
        win-arm64) triple="aarch64-windows-gnu"; lib="sharplight_video.dll" ;;
        osx-arm64) triple="aarch64-macos.11.0"; lib="libsharplight_video.dylib"; link=(-Wl,-install_name,@rpath/libsharplight_video.dylib) ;;
        *) echo "unknown rid $rid" >&2; exit 2 ;;
    esac

    local variant="$rid"
    local opt=(-O2)
    if [ "$sanitize" = 1 ]; then
        variant="$rid-asan"
        opt=(-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
    fi

    local obj="$build_root/$variant"
    local out="$out_root/$variant"
    mkdir -p "$obj" "$out"

    local common=(-target "$triple" -fvisibility=hidden -fPIC -ffunction-sections -fdata-sections "${opt[@]}" -DNDEBUG "${extra[@]}")
    # libvpx is vendored unmodified and compiled without warnings; our own source is held to -Werror.
    local tp_flags=(-w -std=gnu99 -I"$here/config" -I"$tp/libvpx")
    local own_flags=(-std=c11 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wno-sign-conversion -I"$here/include" -I"$here/config" -isystem "$tp/libvpx")
    if [ "$sanitize" = 1 ]; then
        # libvpx relies on shifts of negative values and similar arithmetic that is undefined in C but identical on every compiler we ship.
        tp_flags+=(-fno-sanitize=shift-base,signed-integer-overflow,float-cast-overflow,alignment,object-size)
    fi

    local list="$obj/jobs.txt"
    : > "$list"
    local src
    while read -r src; do
        echo "tp|$tp/$src" >> "$list"
    done < "$here/libvpx-sources.txt"
    echo "tp|$here/config/vpx_config.c" >> "$list"
    for src in "$here"/src/*.c; do
        echo "own|$src" >> "$list"
    done

    export CC_COMMAND="$zig cc" OBJ_DIR="$obj"
    export COMMON_FLAGS="${common[*]}" TP_FLAGS="${tp_flags[*]}" OWN_FLAGS="${own_flags[*]}"
    # The entry is passed as $1 (not substituted with -I): BSD xargs limits -I replacements to 255 bytes.
    tr '\n' '\0' < "$list" | xargs -0 -n 1 -P "$jobs" bash -c '
        entry="$1"; kind="${entry%%|*}"; src="${entry#*|}"
        name="$(echo "$src" | sed "s|/|_|g")"
        o="$OBJ_DIR/$name.o"
        if [ -f "$o" ] && [ "$o" -nt "$src" ] && [ -z "${SLV_FORCE:-}" ]; then exit 0; fi
        if [ "$kind" = tp ]; then flags="$TP_FLAGS"; else flags="$OWN_FLAGS"; fi
        nice -n 15 $CC_COMMAND $COMMON_FLAGS $flags -c "$src" -o "$o" || { echo "FAILED: $src" >&2; exit 255; }
    ' _

    local objs=()
    while read -r src; do
        objs+=("$obj/$(echo "${src#*|}" | sed 's|/|_|g').o")
    done < "$list"

    local exports
    exports="$(grep '^SLV_API' "$here/include/sharplight_video.h" | grep -oE 'slv_[a-z_]+\(' | tr -d '(' | sort -u)"
    local shared=(-shared)
    case "$rid" in
        linux-*)
            shared+=(-Wl,--gc-sections)
            { echo "{ global:"; echo "$exports" | sed 's/$/;/'; echo "local: *; };"; } > "$obj/exports.map"
            shared+=("-Wl,--version-script=$obj/exports.map")
            ;;
        osx-*)
            # No export list: everything is compiled with -fvisibility=hidden and only the SLV_API functions are marked visible.
            ;;
    esac
    if [ "$sanitize" = 1 ]; then
        shared+=(-fsanitize=address,undefined)
    else
        shared+=(-s)
    fi
    nice -n 15 $zig cc -target "$triple" "${shared[@]}" "${objs[@]}" -o "$out/$lib" "${link[@]}"
    rm -f "$out"/*.lib "$out"/*.pdb
    echo "built $out/$lib"
}

case "$target_arg" in
    all)
        for rid in linux-x64 linux-arm64 win-x64 win-arm64 osx-arm64; do
            compile_target "$rid"
        done
        ;;
    host-sanitize) compile_target linux-x64 1 ;;
    *) compile_target "$target_arg" ;;
esac
