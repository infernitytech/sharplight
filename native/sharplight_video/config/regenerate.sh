#!/usr/bin/env bash
# Regenerates the generic-target configuration headers of libvpx (vpx_config.h, vpx_config.c, vpx_version.h and the four *_rtcd.h)
# from the pinned libvpx release in third_party/sources.json. The result is committed in this directory; run this only when the libvpx
# version or its configure options change, then re-apply the Windows thread switch described at the top of vpx_config.h's HAVE_PTHREAD_H
# definition (vpx_config.h is the only file edited by hand) and update ../libvpx-sources.txt from `make -n libvpx.a` of the same build.
#   regenerate.sh <scratch-directory>
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scratch="${1:?scratch directory required}"
fields="$(python3 -I - "$here/../../../third_party/sources.json" <<'PY'
import json, sys
with open(sys.argv[1]) as file:
    for c in json.load(file)["components"]:
        if c["name"] == "libvpx":
            print(c["url"], c["sha256"], c["archiveRoot"])
PY
)"
read -r url sha256 root <<< "$fields"
mkdir -p "$scratch/extract" "$scratch/build"
curl -fsSL --retry 3 -o "$scratch/libvpx.tar.gz" "$url"
test "$(sha256sum "$scratch/libvpx.tar.gz" | cut -d' ' -f1)" = "$sha256" || { echo "SHA-256 mismatch" >&2; exit 1; }
tar -xzf "$scratch/libvpx.tar.gz" --no-same-owner -C "$scratch/extract"
cd "$scratch/build"
"$scratch/extract/$root/configure" --target=generic-gnu --disable-examples --disable-tools --disable-docs --disable-unit-tests \
    --disable-install-docs --disable-vp8-decoder --disable-vp9-decoder --enable-vp8-encoder --enable-vp9-encoder \
    --disable-runtime-cpu-detect --enable-pic --enable-multithread --disable-webm-io --disable-libyuv --disable-postproc \
    --disable-vp9-postproc --disable-temporal-denoising --disable-vp9-temporal-denoising --disable-install-bins \
    --disable-install-libs --disable-install-srcs
make vpx_version.h vp8_rtcd.h vp9_rtcd.h vpx_dsp_rtcd.h vpx_scale_rtcd.h
cp vp8_rtcd.h vp9_rtcd.h vpx_dsp_rtcd.h vpx_scale_rtcd.h vpx_version.h vpx_config.c vpx_config.h "$here/"
