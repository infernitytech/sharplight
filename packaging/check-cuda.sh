#!/usr/bin/env bash
# Checks the CUDA files of a published Studio against the runtime it was published for: linux-x64, linux-arm64 and win-x64 carry the CUDA
# execution provider (Sharplight.Cuda.dll) with the NVRTC redistributables, the floating-point type headers NVRTC compiles against, the
# NVIDIA license and the third-party notice; every other runtime (osx-arm64, win-arm64, ...) carries none of them.
#   check-cuda.sh <publish-directory> <rid>
# The notice and the license are copied by pack-studio.sh before the check (use --files-only to check just the build's own output).
set -euo pipefail

publish="${1:?publish directory required}"
rid="${2:?rid required}"
files_only="${3:-}"
case "$rid" in
    linux-x64|linux-arm64) native=(libnvrtc.so.13 libnvrtc-builtins.so.13.0) ;;
    win-x64) native=(nvrtc64_130_0.dll nvrtc-builtins64_130.dll) ;;
    *) native=() ;;
esac
headers=(cuda_fp16.h cuda_fp16.hpp cuda_bf16.h cuda_bf16.hpp cuda_fp8.h cuda_fp8.hpp)
documents=(THIRD-PARTY-NOTICES-Cuda.md LICENSE-NVIDIA-CUDA.txt)

present() {
    [ -f "$publish/$1" ] || [ -f "$publish/runtimes/$rid/native/$1" ]
}

if [ ${#native[@]} -eq 0 ]; then
    unexpected=()
    for name in Sharplight.Cuda.dll "${headers[@]}" libnvrtc.so.13 libnvrtc-builtins.so.13.0 nvrtc64_130_0.dll nvrtc-builtins64_130.dll "${documents[@]}"; do
        if present "$name"; then
            unexpected+=("$name")
        fi
    done
    if [ ${#unexpected[@]} -gt 0 ]; then
        echo "$rid must not carry the CUDA provider, but $publish has: ${unexpected[*]}" >&2
        exit 1
    fi
    echo "CUDA for $rid: none, as intended"
    exit 0
fi

missing=()
for name in Sharplight.Cuda.dll "${native[@]}" "${headers[@]}"; do
    if ! present "$name"; then
        missing+=("$name")
    fi
done
if [ "$files_only" != "--files-only" ]; then
    for name in "${documents[@]}"; do
        if [ ! -f "$publish/$name" ]; then
            missing+=("$name")
        fi
    done
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "the CUDA files for $rid are missing from $publish: ${missing[*]} (publish with -r $rid; the NVIDIA redistributables are downloaded from developer.download.nvidia.com by Sharplight.Cuda.Runtime)" >&2
    exit 1
fi

echo "CUDA for $rid: the provider, ${#native[@]} NVRTC libraries, ${#headers[@]} headers$([ "$files_only" = "--files-only" ] || echo ', the license and the notice') are in $publish"
