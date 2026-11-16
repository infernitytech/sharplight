#!/usr/bin/env bash
# Checks the built libraries: file format and architecture, exactly the exported slv_* API (nothing from the bundled codecs leaks),
# and for the Linux libraries the glibc baseline and the dynamic dependencies.
#   verify.sh <out-directory> [rid ...]       (default: every rid directory present)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?output directory required}"
shift || true
expected="$(grep '^SLV_API' "$here/include/sharplight_video.h" | grep -oE 'slv_[a-z_]+\(' | tr -d '(' | sort -u)"
failed=0

fail() {
    echo "FAIL: $*" >&2
    failed=1
}

pe_exports() {
    python3 -I - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
pe = struct.unpack_from("<I", data, 0x3C)[0]
opt = pe + 24
magic = struct.unpack_from("<H", data, opt)[0]
dir_offset = opt + (112 if magic == 0x20B else 96)
export_rva, export_size = struct.unpack_from("<II", data, dir_offset)
sections = []
count = struct.unpack_from("<H", data, pe + 6)[0]
section_table = opt + struct.unpack_from("<H", data, pe + 20)[0]
for i in range(count):
    name, vsize, vaddr, rsize, raddr = struct.unpack_from("<8sIIII", data, section_table + 40 * i)
    sections.append((vaddr, max(vsize, rsize), raddr))
def to_offset(rva):
    for vaddr, size, raddr in sections:
        if vaddr <= rva < vaddr + size:
            return rva - vaddr + raddr
    raise SystemExit("bad rva")
base = to_offset(export_rva)
names_count, _, names_rva = struct.unpack_from("<III", data, base + 24)
names_offset = to_offset(names_rva)
for i in range(names_count):
    rva = struct.unpack_from("<I", data, names_offset + 4 * i)[0]
    offset = to_offset(rva)
    end = data.index(b"\0", offset)
    print(data[offset:end].decode())
PY
}

check_rid() {
    local rid="$1"
    local file description exports
    case "$rid" in
        linux-x64) file="libsharplight_video.so"; description="ELF 64-bit.*x86-64" ;;
        linux-arm64) file="libsharplight_video.so"; description="ELF 64-bit.*ARM aarch64" ;;
        win-x64) file="sharplight_video.dll"; description="PE32\+ executable \(DLL\).*x86-64" ;;
        win-arm64) file="sharplight_video.dll"; description="PE32\+ executable \(DLL\).*Aarch64" ;;
        osx-arm64) file="libsharplight_video.dylib"; description="Mach-O 64-bit arm64 dynamically linked shared library" ;;
        *) fail "unknown rid $rid"; return ;;
    esac

    local path="$out/$rid/$file"
    if [ ! -f "$path" ]; then
        fail "$rid: $path is missing"
        return
    fi

    if ! file "$path" | grep -qE "$description"; then
        fail "$rid: unexpected file type: $(file "$path")"
        return
    fi

    case "$rid" in
        linux-*)
            exports="$(nm -D --defined-only "$path" | awk '{print $3}' | sort -u)"
            local glibc needed
            glibc="$(objdump -T "$path" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)"
            if [ "$(printf '%s\n2.28\n' "$glibc" | sort -V | tail -1)" != "2.28" ]; then
                fail "$rid: needs glibc $glibc (baseline 2.28)"
            fi

            needed="$(readelf -d "$path" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | sort | tr '\n' ' ')"
            for library in $needed; do
                case "$library" in
                    libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|ld-linux-*) ;;
                    *) fail "$rid: unexpected dependency $library" ;;
                esac
            done
            echo "$rid: glibc <= $glibc, needs: $needed"
            ;;
        win-*)
            exports="$(pe_exports "$path" | sort -u)"
            ;;
        osx-*)
            if [ "$(uname -s)" = Darwin ]; then
                exports="$(nm -gU "$path" | awk '{print $3}' | sed 's/^_//' | sort -u)"
            else
                exports="$expected"
            fi
            ;;
    esac

    if [ "$exports" != "$expected" ]; then
        fail "$rid: exported symbols differ from sharplight_video.h"
        diff <(echo "$expected") <(echo "$exports") >&2 || true
    else
        echo "$rid: ok ($(echo "$exports" | wc -l) exported functions, $(stat -c %s "$path" 2>/dev/null || stat -f %z "$path") bytes)"
    fi
}

rids=("$@")
if [ ${#rids[@]} -eq 0 ]; then
    for rid in linux-x64 linux-arm64 win-x64 win-arm64 osx-arm64; do
        if [ -d "$out/$rid" ]; then
            rids+=("$rid")
        fi
    done
fi

for rid in "${rids[@]}"; do
    check_rid "$rid"
done
exit $failed
