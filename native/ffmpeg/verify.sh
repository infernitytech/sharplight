#!/usr/bin/env bash
# Checks built FFmpeg libraries before they ship: every library the loader opens is present with the right file type, FFmpeg was
# configured LGPL-only, and the libraries depend on nothing but the operating system and each other (Linux: glibc <= 2.28, no libstdc++,
# libva or other optional system libraries; Windows: system DLLs only, no libwinpthread / libc++ DLLs; macOS: system frameworks only).
#   verify.sh <out-directory> [rid ...]       (default: every rid directory present)
set -euo pipefail

out="${1:?output directory required}"
shift || true
libraries=(avutil swresample swscale avcodec avformat avfilter)
failed=0

fail() {
    echo "FAIL: $*" >&2
    failed=1
}

major() {
    case "$1" in
        avutil) echo 61 ;;
        swresample) echo 7 ;;
        swscale) echo 10 ;;
        avcodec) echo 63 ;;
        avformat) echo 63 ;;
        avfilter) echo 12 ;;
    esac
}

pe_imports() {
    python3 -I - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
pe = struct.unpack_from("<I", data, 0x3C)[0]
opt = pe + 24
magic = struct.unpack_from("<H", data, opt)[0]
dir_offset = opt + (112 if magic == 0x20B else 96)
import_rva, _ = struct.unpack_from("<II", data, dir_offset + 8)
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
offset = to_offset(import_rva)
while True:
    lookup, stamp, chain, name_rva, thunk = struct.unpack_from("<IIIII", data, offset)
    if name_rva == 0:
        break
    start = to_offset(name_rva)
    print(data[start:data.index(b"\0", start)].decode())
    offset += 20
PY
}

check_rid() {
    local rid="$1"
    local description
    case "$rid" in
        linux-x64) description="ELF 64-bit.*x86-64" ;;
        linux-arm64) description="ELF 64-bit.*ARM aarch64" ;;
        win-x64) description="PE32\+ executable \(DLL\).*x86-64" ;;
        win-arm64) description="PE32\+ executable \(DLL\).*Aarch64" ;;
        osx-arm64) description="Mach-O 64-bit.*arm64" ;;
        *) fail "unknown rid $rid"; return ;;
    esac

    local configuration="$out/$rid/ffmpeg-configuration.txt"
    if [ ! -f "$configuration" ] || ! grep -q "^License: LGPL version 2.1 or later" "$configuration"; then
        fail "$rid: $configuration is missing or not LGPL 2.1+"
    fi

    local total=0 library file path size
    for library in "${libraries[@]}"; do
        case "$rid" in
            linux-*) file="lib$library.so.$(major "$library")" ;;
            win-*) file="$library-$(major "$library").dll" ;;
            osx-*) file="lib$library.$(major "$library").dylib" ;;
        esac

        path="$out/$rid/$file"
        if [ ! -f "$path" ]; then
            fail "$rid: $file is missing"
            continue
        fi

        if command -v file > /dev/null && ! file "$path" | grep -qE "$description"; then
            fail "$rid: unexpected file type: $(file "$path")"
        fi

        size="$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path")"
        total=$((total + size))
        case "$rid" in
            linux-*)
                local glibc dependency
                glibc="$(objdump -T "$path" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)"
                if [ -n "$glibc" ] && [ "$(printf '%s\n2.28\n' "$glibc" | sort -V | tail -1)" != "2.28" ]; then
                    fail "$rid: $file needs glibc $glibc (baseline 2.28)"
                fi

                for dependency in $(readelf -d "$path" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/'); do
                    case "$dependency" in
                        libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|ld-linux-*|libavutil.so.*|libswresample.so.*|libswscale.so.*|libavcodec.so.*|libavformat.so.*) ;;
                        *) fail "$rid: $file depends on $dependency" ;;
                    esac
                done
                ;;
            win-*)
                local import
                for import in $(pe_imports "$path" | tr 'A-Z' 'a-z'); do
                    case "$import" in
                        kernel32.dll|user32.dll|advapi32.dll|bcrypt.dll|ole32.dll|oleaut32.dll|shell32.dll|ws2_32.dll|secur32.dll|gdi32.dll|psapi.dll|mfplat.dll|mfuuid.dll|strmiids.dll|api-ms-win-*.dll|avutil-*.dll|swresample-*.dll|swscale-*.dll|avcodec-*.dll|avformat-*.dll) ;;
                        *) fail "$rid: $file imports $import" ;;
                    esac
                done
                ;;
            osx-*)
                if [ "$(uname -s)" = Darwin ]; then
                    local dependency
                    for dependency in $(otool -L "$path" | tail -n +2 | awk '{print $1}'); do
                        case "$dependency" in
                            /System/Library/Frameworks/*|/usr/lib/libSystem.B.dylib|/usr/lib/libc++.1.dylib|/usr/lib/libz.1.dylib|/usr/lib/libiconv.2.dylib|/usr/lib/libobjc.A.dylib|@rpath/libav*|@rpath/libsw*) ;;
                            *) fail "$rid: $file depends on $dependency" ;;
                        esac
                    done
                fi
                ;;
        esac
    done
    echo "$rid: $(du -sh "$out/$rid" | cut -f1) in $out/$rid ($((total / 1024)) KiB of libraries)"
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
