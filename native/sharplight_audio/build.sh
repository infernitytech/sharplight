#!/usr/bin/env bash
# Cross-compiles the Sharplight audio library with zig cc.
#   build.sh <rid|all|host-sanitize> [--jobs N]
# RIDs: linux-x64 linux-arm64 win-x64 win-arm64 (cross-compiled with zig on any host) and osx-arm64 (needs the Apple SDK for the
# CoreAudio / AudioToolbox headers miniaudio includes, so it is built with the system clang on a macOS host only). "host-sanitize" builds linux-x64 with ASan+UBSan and the C test.
# Environment: ZIG (path to zig, default: zig on PATH), SLA_OUT (output root, default native/sharplight_audio/out),
#              SLA_BUILD (object cache, default $SLA_OUT/obj).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
tp="$root/third_party"
zig="${ZIG:-zig}"
out_root="${SLA_OUT:-$here/out}"
build_root="${SLA_BUILD:-$out_root/obj}"
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

third_party_sources() {
    local f
    for f in celt/bands celt/celt celt/celt_encoder celt/celt_decoder celt/cwrs celt/entcode celt/entdec celt/entenc \
        celt/kiss_fft celt/laplace celt/mathops celt/mdct celt/modes celt/pitch celt/celt_lpc celt/quant_bands celt/rate celt/vq \
        src/opus src/opus_decoder src/opus_encoder src/extensions src/opus_multistream src/opus_multistream_encoder \
        src/opus_multistream_decoder src/repacketizer src/opus_projection_encoder src/opus_projection_decoder \
        src/mapping_matrix src/analysis src/mlp src/mlp_data; do
        echo "$tp/opus/$f.c"
    done
    ls "$tp"/opus/silk/*.c "$tp"/opus/silk/float/*.c
    echo "$tp/ogg/src/bitwise.c"
    echo "$tp/ogg/src/framing.c"
    for f in info internal opusfile stream; do
        echo "$tp/opusfile/src/$f.c"
    done
    for f in analysis bitrate block codebook envelope floor0 floor1 info lookup lpc lsp mapping0 mdct psy registry res0 sharedbook smallft \
        synthesis vorbisenc vorbisfile window; do
        echo "$tp/vorbis/lib/$f.c"
    done
}

our_sources() {
    ls "$here"/src/*.c | grep -v miniaudio_impl.c
}

compile_target() {
    local rid="$1"
    local sanitize="${2:-0}"
    local triple lib link=() native_clang=0
    case "$rid" in
        linux-x64) triple="x86_64-linux-gnu.2.28"; lib="libsharplight_audio.so"; link=(-ldl -lpthread -lm) ;;
        linux-arm64) triple="aarch64-linux-gnu.2.28"; lib="libsharplight_audio.so"; link=(-ldl -lpthread -lm) ;;
        win-x64) triple="x86_64-windows-gnu"; lib="sharplight_audio.dll" ;;
        win-arm64) triple="aarch64-windows-gnu"; lib="sharplight_audio.dll" ;;
        osx-arm64)
            if [ "$(uname -s)" != Darwin ]; then
                echo "osx-arm64 needs the Apple SDK (CoreAudio headers); build it on a macOS host" >&2
                return 0
            fi
            triple="arm64-apple-macos11.0"; lib="libsharplight_audio.dylib"; native_clang=1
            link=(-dynamiclib -Wl,-install_name,@rpath/libsharplight_audio.dylib -framework CoreFoundation -framework CoreAudio -framework AudioToolbox)
            ;;
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

    local cc_command="$zig cc"
    if [ "$native_clang" = 1 ]; then
        cc_command="clang"
    fi

    local common=(-target "$triple" -std=c11 -fvisibility=hidden -fPIC -ffunction-sections -fdata-sections "${opt[@]}" -DNDEBUG)
    local tp_flags=(-w -DOPUS_BUILD -DVAR_ARRAYS -std=gnu11 -DHAVE_LRINT -DHAVE_LRINTF "-DPACKAGE_VERSION=\"1.6.1\"" -DENABLE_HARDENING -DOP_DISABLE_HTTP -DOP_DISABLE_DOCS
        -I"$tp/opus/include" -I"$tp/opus/celt" -I"$tp/opus/silk" -I"$tp/opus/silk/float" -I"$tp/opus/src"
        -I"$here/config" -I"$tp/ogg/include" -I"$tp/opusfile/include" -I"$tp/opusfile/src" -I"$tp/vorbis/include" -I"$tp/vorbis/lib" -I"$tp/miniaudio" -I"$tp/miniaudio/extras")
    local own_flags=(-Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wno-sign-conversion
        -I"$here/include" -isystem "$tp/miniaudio" -isystem "$tp/miniaudio/extras" -isystem "$tp/opus/include" -isystem "$tp/ogg/include" -I"$here/config" -isystem "$tp/opusfile/include" -isystem "$tp/vorbis/include")

    if [ "$sanitize" = 1 ]; then
        # The vendored codecs rely on arithmetic that is undefined in C but behaves the same on every compiler we ship with (shifting
        # negative values, float-to-integer casts of NaN while seeking in corrupt files, signed wrap-around). They are not our code and are
        # not patched, so those three UBSan checks are off for them; address checking and all other checks stay on, and our own sources
        # get every check.
        tp_flags+=(-fno-sanitize=float-cast-overflow,shift-base,signed-integer-overflow)
    fi

    local list="$obj/jobs.txt"
    : > "$list"
    local src
    while read -r src; do
        echo "tp|$src" >> "$list"
    done < <(third_party_sources)
    echo "tp|$here/src/miniaudio_impl.c" >> "$list"
    while read -r src; do
        echo "own|$src" >> "$list"
    done < <(our_sources)

    export CC_COMMAND="$cc_command" OBJ_DIR="$obj"
    export COMMON_FLAGS="${common[*]}" TP_FLAGS="${tp_flags[*]}" OWN_FLAGS="${own_flags[*]}"
    # shellcheck disable=SC2016
    # The entry is passed as $1 (not substituted with -I): BSD xargs on macOS limits -I replacements to 255 bytes.
    tr '\n' '\0' < "$list" | xargs -0 -n 1 -P "$jobs" bash -c '
        entry="$1"; kind="${entry%%|*}"; src="${entry#*|}"
        name="$(echo "$src" | sed "s|/|_|g")"
        o="$OBJ_DIR/$name.o"
        if [ -f "$o" ] && [ "$o" -nt "$src" ] && [ -z "$SLA_FORCE" ]; then exit 0; fi
        if [ "$kind" = tp ]; then flags="$TP_FLAGS"; else flags="$OWN_FLAGS"; fi
        nice -n 15 $CC_COMMAND $COMMON_FLAGS $flags -c "$src" -o "$o" || { echo "FAILED: $src" >&2; exit 255; }
    ' _

    local objs=()
    while read -r src; do
        objs+=("$obj/$(echo "${src#*|}" | sed 's|/|_|g').o")
    done < "$list"

    local exports
    exports="$(grep '^SLA_API' "$here/include/sharplight_audio.h" | grep -oE 'sla_[a-z_]+\(' | tr -d '(' | sort -u)"
    local shared=(-shared)
    case "$rid" in
        linux-*)
            shared+=(-Wl,--gc-sections)
            { echo "{ global:"; echo "$exports" | sed 's/$/;/'; echo "local: *; };"; } > "$obj/exports.map"
            shared+=("-Wl,--version-script=$obj/exports.map")
            ;;
        osx-*)
            echo "$exports" | sed 's/^/_/' > "$obj/exports.txt"
            shared=("-Wl,-exported_symbols_list,$obj/exports.txt")
            ;;
    esac
    if [ "$sanitize" = 1 ]; then
        shared+=(-fsanitize=address,undefined)
    elif [ "$native_clang" != 1 ]; then
        shared+=(-s)
    fi
    if [ "$native_clang" = 1 ]; then
        nice -n 15 clang -target "$triple" "${shared[@]}" "${objs[@]}" -o "$out/$lib" "${link[@]}"
        strip -x "$out/$lib"
    else
        nice -n 15 $cc_command -target "$triple" "${shared[@]}" "${objs[@]}" -o "$out/$lib" "${link[@]}"
    fi
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
