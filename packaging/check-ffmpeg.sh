#!/usr/bin/env bash
# Fails when a published application lacks the bundled FFmpeg libraries or their license texts (Sharplight.Video copies them to
# runtimes/<rid>/native/ and licenses/ffmpeg/; a NuGet consumer's publish flattens the libraries into the application directory).
#   check-ffmpeg.sh <publish-directory> <rid>
set -euo pipefail

publish="${1:?publish directory required}"
rid="${2:?rid required}"
case "$rid" in
    linux-*) names=(libavutil.so.61 libswresample.so.7 libswscale.so.10 libavcodec.so.63 libavformat.so.63 libavfilter.so.12) ;;
    win-*) names=(avutil-61.dll swresample-7.dll swscale-10.dll avcodec-63.dll avformat-63.dll avfilter-12.dll) ;;
    osx-*) names=(libavutil.61.dylib libswresample.7.dylib libswscale.10.dylib libavcodec.63.dylib libavformat.63.dylib libavfilter.12.dylib) ;;
    *) echo "unknown rid $rid" >&2; exit 2 ;;
esac

missing=()
for name in "${names[@]}"; do
    if [ ! -f "$publish/runtimes/$rid/native/$name" ] && [ ! -f "$publish/$name" ]; then
        missing+=("$name")
    fi
done

if [ ! -f "$publish/licenses/ffmpeg/FFmpeg-COPYING.LGPLv2.1.txt" ]; then
    missing+=("licenses/ffmpeg/FFmpeg-COPYING.LGPLv2.1.txt")
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo "the bundled FFmpeg files for $rid are missing from $publish: ${missing[*]} (build them with native/ffmpeg/build.sh or download the native-ffmpeg-* artifacts into native/ffmpeg/out)" >&2
    exit 1
fi

echo "FFmpeg for $rid: ${#names[@]} libraries and the license texts are in $publish"
