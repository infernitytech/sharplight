#!/usr/bin/env bash
# Driver-only check of the CUDA provider: runs a self-contained published app (e.g. the CLI, whose directory holds
# Sharplight.Cuda.Runtime's NVRTC and headers) so that nothing of a CUDA toolkit can be found:
#   - an empty environment (no PATH entries of a toolkit, no LD_LIBRARY_PATH, no CUDA_PATH / CUDA_HOME),
#   - the dynamic loader without its cache (ld.so --inhibit-cache): /etc/ld.so.conf entries such as /usr/local/cuda/lib64 or the
#     toolkit's forward-compatibility driver (/usr/local/cuda-*/compat) are not searched, only the loader's built-in system
#     directories, which hold glibc and the NVIDIA driver's user-space libraries (libcuda.so.1, libnvidia-ptxjitcompiler.so.1);
#     hostfxr-launch.c starts the app with explicit paths, since under ld.so the apphost cannot find its own directory,
#   - SHARPLIGHT_CUDA_RUNTIME_DIR unset, so NVRTC can only come from the app directory,
#   - a new, empty HOME: no per-user kernel cache, so every kernel is compiled (Slang, NVRTC, the driver's JIT) in the run.
# Afterwards it lists every CUDA / NVIDIA library the process had mapped and fails if one came from a toolkit directory.
#
# Containers allowing it can do the same more strictly: `docker run --gpus all -v <publish dir>:/app ubuntu:24.04 /app/Sharplight.Cli ...`
# (the NVIDIA container toolkit mounts only the driver's libraries and device nodes into the image).
#
# Usage: cuda-driver-only.sh <published app dir> <model file> [app arguments...]
set -euo pipefail

app=$(cd "$1" && pwd)
model=$2
shift 2
maps=$(mktemp)
launcher=$(mktemp)
home=$(mktemp -d)
cc -O2 -o "$launcher" "$(dirname "$0")/hostfxr-launch.c" -ldl

env -i HOME="$home" TMPDIR=/tmp DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    /lib64/ld-linux-x86-64.so.2 --inhibit-cache "$launcher" "$app" Sharplight.Cli "$model" "$@" &
pid=$!
# Sample the mapped libraries while the app runs (it loads NVRTC and the driver when it compiles the first kernels).
while kill -0 "$pid" 2>/dev/null; do
    grep -oE '/[^ ]*\.so[.0-9]*' "/proc/$pid/maps" 2>/dev/null >> "$maps" || true
    sleep 1
done
status=0
wait "$pid" || status=$?

echo "CUDA / NVIDIA libraries the process mapped:"
grep -E 'lib(cuda|nvrtc|nvidia-ptxjit|nvidia-gpucomp|nvidia-nvvm|cudart|cublas)' "$maps" | sort -u | sed 's/^/  /'
if grep -E '^/usr/local/cuda|/targets/x86_64-linux/|/compat/' "$maps" | grep -q .; then
    echo "FAIL: a CUDA toolkit library was loaded"
    exit 1
fi

echo "OK: NVRTC from the app directory, the driver API from the driver installation"
rm -rf "$maps" "$launcher" "$home"
exit "$status"
