#!/usr/bin/env bash
# Builds Sharplight's own FFmpeg: LGPL-only shared libraries (avutil, swscale, swresample, avcodec, avformat) with the codec libraries
# linked in statically (libvpx, dav1d, SVT-AV1, OpenH264, libopus, zlib; all BSD/zlib licensed), so the application can ship them
# instead of relying on a system ffmpeg. Never configured with --enable-gpl or --enable-nonfree.
#   build.sh <rid> [--jobs N]
# RIDs and toolchains:
#   linux-x64, linux-arm64   zig cc (glibc 2.28 baseline, libc++ linked statically), on any x86_64 Linux host
#   win-x64, win-arm64       llvm-mingw (UCRT; libc++ and libunwind linked statically), on an x86_64 Linux host
#   osx-arm64                the system clang on a macOS host (Apple SDK for VideoToolbox / AudioToolbox), deployment target 11.0
# Sources and build tools are pinned in third_party/sources.json ("buildSources", "buildTools") and fetched by fetch.sh.
# Host requirements: python3, curl, make, cmake, ninja, pkg-config, a host C compiler (for nasm), perl; meson is installed into a
# venv when it is not on PATH.
# Environment: ZIG (zig binary for the Linux targets), SLF_CACHE (downloads + extracted sources, default native/ffmpeg/cache),
#              SLF_BUILD (build trees, default native/ffmpeg/build), SLF_OUT (output root, default native/ffmpeg/out; the libraries
#              land in $SLF_OUT/<rid>/), SLF_KEEP (keep the build tree of a successful build).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
rid="${1:?rid required (linux-x64, linux-arm64, win-x64, win-arm64, osx-arm64)}"
shift
jobs=4
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs) jobs="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

cache="${SLF_CACHE:-$here/cache}"
build="${SLF_BUILD:-$here/build}/$rid"
out="${SLF_OUT:-$here/out}/$rid"
prefix="$build/prefix"
tc="$build/toolchain"
mkdir -p "$build" "$prefix/lib/pkgconfig" "$prefix/include" "$tc" "$out"

sources="$("$here/fetch.sh" "$cache")"
# The extracted directory of a fetched component (bash 3 compatible: macOS ships bash 3.2).
source_dir() {
    echo "$sources" | awk -v name="$1" '$1 == name { print $2 }'
}

# The license texts shipped with the libraries (native/ffmpeg/licenses) must be those of the pinned sources.
licenses="$here/licenses"
check_license() {
    if ! cmp -s "$1" "$licenses/$2"; then
        echo "native/ffmpeg/licenses/$2 differs from $1 of the pinned source; update it" >&2
        exit 1
    fi
}
check_license "$(source_dir ffmpeg)/COPYING.LGPLv2.1" FFmpeg-COPYING.LGPLv2.1.txt
check_license "$(source_dir libvpx)/LICENSE" libvpx-LICENSE.txt
check_license "$(source_dir libvpx)/PATENTS" libvpx-PATENTS.txt
check_license "$(source_dir dav1d)/COPYING" dav1d-COPYING.txt
check_license "$(source_dir svt-av1)/LICENSE.md" SVT-AV1-LICENSE.md
check_license "$(source_dir svt-av1)/PATENTS.md" SVT-AV1-PATENTS.md
check_license "$(source_dir openh264)/LICENSE" OpenH264-LICENSE.txt
check_license "$(source_dir opus)/COPYING" opus-COPYING.txt
check_license "$(source_dir zlib)/LICENSE" zlib-LICENSE.txt
check_license "$(source_dir llvm-mingw)/LICENSE.TXT" LLVM-LICENSE.txt
check_license "$(source_dir llvm-mingw)/x86_64-w64-mingw32/share/mingw32/COPYING.MinGW-w64-runtime.txt" MinGW-w64-runtime-COPYING.txt

host_os="$(uname -s)"
run() {
    if [ "$host_os" = Linux ]; then
        nice -n 15 ionice -c3 "$@"
    else
        nice -n 15 "$@"
    fi
}

# ---------------------------------------------------------------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------------------------------------------------------------
x86=0
case "$rid" in
    linux-x64) family=linux; arch=x86_64; zig_triple="x86_64-linux-gnu.2.28"; x86=1 ;;
    linux-arm64) family=linux; arch=aarch64; zig_triple="aarch64-linux-gnu.2.28" ;;
    win-x64) family=windows; arch=x86_64; x86=1 ;;
    win-arm64) family=windows; arch=aarch64 ;;
    osx-arm64) family=darwin; arch=aarch64 ;;
    *) echo "unknown rid $rid" >&2; exit 2 ;;
esac

opt_flags="-O2 -fPIC -ffunction-sections -fdata-sections"
cxx_runtime=""
case "$family" in
    linux)
        zig="${ZIG:-zig}"
        export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$build/zig-global}"
        export ZIG_LOCAL_CACHE_DIR="${ZIG_LOCAL_CACHE_DIR:-$build/zig-local}"
        # zig cc enables UBSan traps in unoptimized code; configure probes are compiled that way, the real objects with -O2.
        printf '#!/bin/sh\nexec "%s" cc -target %s -fno-sanitize=all "$@"\n' "$zig" "$zig_triple" > "$tc/cc"
        printf '#!/bin/sh\nexec "%s" c++ -target %s -fno-sanitize=all "$@"\n' "$zig" "$zig_triple" > "$tc/c++"
        # Meson asks for thin archives (the T modifier), which zig's linker cannot read: write normal ones.
        printf '#!/bin/sh\nflags="$1"\nshift\ncase "$flags" in -*) ;; *) flags="$(echo "$flags" | tr -d T)" ;; esac\nexec "%s" ar "$flags" "$@"\n' "$zig" > "$tc/ar"
        printf '#!/bin/sh\nexec "%s" ranlib "$@"\n' "$zig" > "$tc/ranlib"
        printf '#!/bin/sh\nexit 0\n' > "$tc/strip"
        chmod +x "$tc"/*
        CC="$tc/cc"; CXX="$tc/c++"; AR="$tc/ar"; RANLIB="$tc/ranlib"; NM="nm"; STRIP="$tc/strip"; WINDRES=""
        cmake_system=Linux; meson_system=linux
        cxx_runtime="-lc++"
        ;;
    windows)
        mingw="$(source_dir llvm-mingw)/bin"
        cross="$arch-w64-mingw32-"
        CC="$mingw/${cross}clang"; CXX="$mingw/${cross}clang++"; AR="$mingw/${cross}ar"; RANLIB="$mingw/${cross}ranlib"
        NM="$mingw/${cross}nm"; STRIP="$mingw/${cross}strip"; WINDRES="$mingw/${cross}windres"
        cmake_system=Windows; meson_system=windows
        opt_flags="-O2 -ffunction-sections -fdata-sections"
        # The C++ runtime of OpenH264 is linked into avcodec, so no libc++.dll / libunwind.dll has to ship.
        cxx_runtime="-Wl,-Bstatic -lc++ -lunwind -Wl,-Bdynamic"
        ;;
    darwin)
        CC="$(xcrun -f clang)"; CXX="$(xcrun -f clang++)"; AR="$(xcrun -f ar)"; RANLIB="$(xcrun -f ranlib)"; NM="$(xcrun -f nm)"
        STRIP="$(xcrun -f strip)"; WINDRES=""
        export MACOSX_DEPLOYMENT_TARGET=11.0
        opt_flags="$opt_flags -arch arm64 -mmacosx-version-min=11.0"
        cmake_system=Darwin; meson_system=darwin
        cxx_runtime="-lc++"
        ;;
esac

# nasm for the x86 assembly of FFmpeg, dav1d, SVT-AV1, OpenH264 and libvpx (a build tool, nothing of it ships).
nasm=""
if [ "$x86" = 1 ]; then
    nasm="$cache/tools/nasm/bin/nasm"
    if [ ! -x "$nasm" ]; then
        rm -rf "$build/nasm"
        cp -r "$(source_dir nasm)" "$build/nasm"
        (cd "$build/nasm" && run ./configure --prefix="$cache/tools/nasm" > configure.log && run make -j"$jobs" nasm > make.log && mkdir -p "$cache/tools/nasm/bin" && cp nasm "$nasm")
    fi
fi

if command -v meson > /dev/null; then
    meson="meson"
else
    if [ ! -x "$cache/tools/venv/bin/meson" ]; then
        python3 -m venv "$cache/tools/venv"
        printf 'meson==1.12.1 --hash=sha256:930bc7542cbd9f57009e182fd014eba48cf1a0180a7b9006d2d32d8f168d8b02\n' > "$cache/tools/meson-requirements.txt"
        "$cache/tools/venv/bin/pip" install --quiet --require-hashes --only-binary :all: -r "$cache/tools/meson-requirements.txt"
    fi
    meson="$cache/tools/venv/bin/meson"
fi

export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
export PKG_CONFIG_PATH=""

cmake_toolchain="$tc/toolchain.cmake"
{
    if [ "$family" != darwin ]; then
        echo "set(CMAKE_SYSTEM_NAME $cmake_system)"
        echo "set(CMAKE_SYSTEM_PROCESSOR $arch)"
        echo "set(CMAKE_C_COMPILER \"$CC\")"
        echo "set(CMAKE_CXX_COMPILER \"$CXX\")"
        echo "set(CMAKE_AR \"$AR\" CACHE FILEPATH \"\")"
        echo "set(CMAKE_RANLIB \"$RANLIB\" CACHE FILEPATH \"\")"
        if [ -n "$WINDRES" ]; then
            echo "set(CMAKE_RC_COMPILER \"$WINDRES\")"
        fi
        # Checks only compile: linking a probe would make zig build libc++ again for every CPU feature flag tested.
        echo "set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)"
        echo "set(CMAKE_FIND_ROOT_PATH \"$prefix\")"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)"
    else
        echo "set(CMAKE_OSX_ARCHITECTURES arm64)"
        echo "set(CMAKE_OSX_DEPLOYMENT_TARGET 11.0)"
    fi
    if [ -n "$nasm" ]; then
        echo "set(CMAKE_ASM_NASM_COMPILER \"$nasm\")"
    fi
} > "$cmake_toolchain"

meson_cross="$tc/meson.ini"
{
    echo "[binaries]"
    echo "c = '$CC'"
    echo "cpp = '$CXX'"
    echo "ar = '$AR'"
    echo "strip = '$STRIP'"
    echo "pkg-config = 'pkg-config'"
    if [ -n "$nasm" ]; then
        echo "nasm = '$nasm'"
    fi
    if [ -n "$WINDRES" ]; then
        echo "windres = '$WINDRES'"
    fi
    echo "[built-in options]"
    echo "c_args = [$(printf "'%s', " $opt_flags)]"
    echo "cpp_args = [$(printf "'%s', " $opt_flags)]"
    if [ "$family" = darwin ]; then
        echo "c_link_args = ['-arch', 'arm64', '-mmacosx-version-min=11.0']"
        echo "cpp_link_args = ['-arch', 'arm64', '-mmacosx-version-min=11.0']"
    fi
    echo "[host_machine]"
    echo "system = '$meson_system'"
    echo "cpu_family = '$arch'"
    echo "cpu = '$arch'"
    echo "endian = 'little'"
} > "$meson_cross"
if [ "$family" = darwin ]; then
    meson_target=(--native-file "$meson_cross")
else
    meson_target=(--cross-file "$meson_cross")
fi

cmake_build() {
    local name="$1"
    shift
    local dir="$build/$name"
    rm -rf "$dir"
    run cmake -S "$(source_dir "$name")" -B "$dir" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$cmake_toolchain" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_FLAGS="$opt_flags" -DCMAKE_CXX_FLAGS="$opt_flags" "$@" > "$build/$name.configure.log"
    run cmake --build "$dir" --parallel "$jobs" > "$build/$name.build.log"
    run cmake --install "$dir" > "$build/$name.install.log"
}

meson_build() {
    local name="$1"
    shift
    local dir="$build/$name"
    rm -rf "$dir"
    run "$meson" setup "$dir" "$(source_dir "$name")" "${meson_target[@]}" --prefix "$prefix" --libdir lib --buildtype release \
        --default-library static "$@" > "$build/$name.configure.log"
    run ninja -C "$dir" -j "$jobs" > "$build/$name.build.log"
    run ninja -C "$dir" install > "$build/$name.install.log"
}

step() {
    echo "== $rid: $1"
}

# Finished dependency steps leave a stamp, so a rerun with SLF_KEEP (after a failure further on) resumes instead of rebuilding.
stamps="$build/stamps"
mkdir -p "$stamps"
pending() {
    if [ -f "$stamps/$1" ]; then
        echo "== $rid: $1 (done)"
        return 1
    fi

    step "$1"
}

done_step() {
    touch "$stamps/$1"
}

# ---------------------------------------------------------------------------------------------------------------------------------
# Codec libraries (static)
# ---------------------------------------------------------------------------------------------------------------------------------
if pending zlib; then
mkdir -p "$build/zlib"
zlib_objects=()
for file in adler32 compress crc32 deflate infback inffast inflate inftrees trees uncompr zutil; do
    run "$CC" $opt_flags -w -c "$(source_dir zlib)/$file.c" -o "$build/zlib/$file.o"
    zlib_objects+=("$build/zlib/$file.o")
done
rm -f "$prefix/lib/libz.a"
run "$AR" rcs "$prefix/lib/libz.a" "${zlib_objects[@]}"
cp "$(source_dir zlib)/zlib.h" "$(source_dir zlib)/zconf.h" "$prefix/include/"
done_step zlib
fi

if pending opus; then
opus_args=()
if [ "$arch" = aarch64 ]; then
    # NEON is part of AArch64, and opus has no run-time detection for mingw on Arm; its NEON prototypes are only declared with the
    # MAY_HAVE macros, which CMake leaves out once run-time detection is off.
    opus_args+=(-DOPUS_PRESUME_NEON=ON -DOPUS_MAY_HAVE_NEON=OFF "-DCMAKE_C_FLAGS=$opt_flags -DOPUS_ARM_MAY_HAVE_NEON -DOPUS_ARM_MAY_HAVE_NEON_INTR")
fi
cmake_build opus -DOPUS_BUILD_SHARED_LIBRARY=OFF -DOPUS_BUILD_TESTING=OFF -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_FORTIFY_SOURCE=OFF \
    -DOPUS_STACK_PROTECTOR=OFF -DOPUS_INSTALL_PKG_CONFIG_MODULE=ON -DOPUS_INSTALL_CMAKE_CONFIG_MODULE=OFF ${opus_args[@]+"${opus_args[@]}"}
done_step opus
fi

if pending libvpx; then
vpx_args=(--prefix="$prefix" --libdir="$prefix/lib" --enable-static --disable-shared --enable-pic --disable-examples --disable-tools
    --disable-docs --disable-unit-tests --disable-decode-perf-tests --disable-encode-perf-tests --disable-vp8-decoder --disable-vp9-decoder
    --enable-vp9-highbitdepth --disable-install-bins --disable-install-srcs)
case "$rid" in
    linux-x64) vpx_args+=(--target=x86_64-linux-gcc --as=nasm) ;;
    linux-arm64) vpx_args+=(--target=arm64-linux-gcc) ;;
    win-x64) vpx_args+=(--target=x86_64-win64-gcc --as=nasm) ;;
    win-arm64) vpx_args+=(--target=arm64-win64-gcc) ;;
    osx-arm64) vpx_args+=(--target=arm64-darwin20-gcc) ;;
esac
rm -rf "$build/libvpx"
mkdir -p "$build/libvpx"
(
    cd "$build/libvpx"
    export CC CXX AR NM STRIP LD="$CXX" CFLAGS="$opt_flags" CXXFLAGS="$opt_flags"
    if [ -n "$nasm" ]; then
        export PATH="$(dirname "$nasm"):$PATH"
    fi
    run "$(source_dir libvpx)/configure" "${vpx_args[@]}" > "$build/libvpx.configure.log"
    run make -j"$jobs" > "$build/libvpx.build.log" 2>&1
    run make install > "$build/libvpx.install.log" 2>&1
)
done_step libvpx
fi

if pending dav1d; then
meson_build dav1d -Denable_tools=false -Denable_tests=false -Denable_examples=false -Denable_docs=false
done_step dav1d
fi

if pending svt-av1; then
svt_args=(-DBUILD_APPS=OFF -DBUILD_TESTING=OFF -DSVT_AV1_LTO=OFF -DREPRODUCIBLE_BUILDS=ON -DEXCLUDE_HASH=ON -DENABLE_STACK_PROTECTOR=OFF)
if [ "$family" = windows ]; then
    # Windows on Arm has no SVE support to dispatch to.
    svt_args+=(-DENABLE_SVE=OFF -DENABLE_SVE2=OFF)
fi
if [ "$family" = linux ]; then
    # zig hands clang an explicit CPU feature list without evex512, so -mavx512f alone cannot compile SVT's 512-bit intrinsics; the
    # AVX2 kernels are used instead.
    svt_args+=(-DENABLE_AVX512=OFF)
fi
cmake_build svt-av1 "${svt_args[@]}"
done_step svt-av1
fi

if pending openh264; then
meson_build openh264 -Dtests=disabled
done_step openh264
fi

# The generated .pc files name the GNU C++ runtime and pthreads; point them at the statically linked LLVM runtime of the toolchain,
# and drop pthreads on Windows (every library there uses Win32 threads; linking it would add a libwinpthread-1.dll dependency).
for pc in "$prefix"/lib/pkgconfig/*.pc; do
    sed -i.bak -e "s#-lstdc++#$cxx_runtime#g" "$pc"
    if [ "$family" = windows ]; then
        sed -i.bak -e 's#-lpthread##g' -e 's#-pthread##g' "$pc"
    fi
    rm -f "$pc.bak"
done

# ---------------------------------------------------------------------------------------------------------------------------------
# FFmpeg (shared)
# ---------------------------------------------------------------------------------------------------------------------------------
step "nv-codec-headers"
if [ "$family" != darwin ]; then
    # Copied first: its install target writes the generated .pc into the source tree, which the RIDs share.
    rm -rf "$build/nv-codec-headers"
    cp -r "$(source_dir nv-codec-headers)" "$build/nv-codec-headers"
    make -C "$build/nv-codec-headers" PREFIX="$prefix" install > "$build/nv-codec-headers.log"
fi

step ffmpeg
join() {
    local IFS=,
    echo "$*"
}

decoders=(h264 hevc vp8 vp9 av1 libdav1d mpeg4 mpeg1video mpeg2video h263 mjpeg png apng gif webp prores ffv1 theora rawvideo dnxhd
    qtrle utvideo bmp tiff vc1 wmv3 msmpeg4v3 cfhd
    aac aac_latm mp3 mp3float mp2 opus vorbis flac alac ac3 eac3 pcm_s16le pcm_s16be pcm_s24le pcm_s24be pcm_s32le pcm_f32le pcm_f64le
    pcm_u8 pcm_alaw pcm_mulaw)
encoders=(libopenh264 libsvtav1 libvpx_vp8 libvpx_vp9 prores prores_ks ffv1 mjpeg png apng gif rawvideo aac libopus flac pcm_s16le)
muxers=(mp4 mov ipod webm matroska gif apng image2 null)
demuxers=(mov matroska avi gif apng image2 image_png_pipe image_jpeg_pipe image_webp_pipe mpegts mpegps flv ogg wav mp3 aac flac h264
    hevc ivf m4v obu)
parsers=(h264 hevc vp8 vp9 av1 mpeg4video mpegvideo mjpeg png gif webp aac aac_latm opus vorbis flac mpegaudio ac3 dnxhd vc1)
bsfs=(aac_adtstoasc extract_extradata h264_mp4toannexb hevc_mp4toannexb vp9_superframe vp9_superframe_split av1_frame_merge av1_frame_split
    null)
# The palette filters make GIF; the rest are the plumbing of a filter graph.
filters=(buffer buffersink abuffer abuffersink format aformat scale fps split palettegen paletteuse null anull aresample pad setsar)
hw=()
case "$family" in
    linux)
        hw=(--enable-ffnvcodec --enable-nvenc --enable-nvdec --enable-cuda --enable-cuvid)
        encoders+=(h264_nvenc hevc_nvenc av1_nvenc)
        ;;
    windows)
        hw=(--enable-ffnvcodec --enable-nvenc --enable-nvdec --enable-cuda --enable-cuvid --enable-mediafoundation --enable-d3d11va
            --enable-dxva2 --disable-pthreads --enable-w32threads)
        encoders+=(h264_nvenc hevc_nvenc av1_nvenc h264_mf hevc_mf av1_mf)
        ;;
    darwin)
        hw=(--enable-videotoolbox --enable-audiotoolbox)
        encoders+=(h264_videotoolbox hevc_videotoolbox prores_videotoolbox aac_at)
        decoders+=(aac_at)
        ;;
esac

ff_args=(--prefix="$prefix" --enable-shared --disable-static --disable-programs --disable-doc --disable-avdevice
    --disable-network --disable-autodetect --disable-debug --disable-stripping --enable-pic --enable-zlib
    --enable-libvpx --enable-libdav1d --enable-libsvtav1 --enable-libopenh264 --enable-libopus
    --disable-everything --enable-swscale --enable-swresample --enable-avfilter --enable-filter="$(join "${filters[@]}")"
    --enable-protocol=file,pipe --enable-hwaccels
    --enable-decoder="$(join "${decoders[@]}")" --enable-encoder="$(join "${encoders[@]}")" --enable-muxer="$(join "${muxers[@]}")"
    --enable-demuxer="$(join "${demuxers[@]}")" --enable-parser="$(join "${parsers[@]}")" --enable-bsf="$(join "${bsfs[@]}")"
    --pkg-config=pkg-config --pkg-config-flags=--static
    --extra-cflags="-I$prefix/include $opt_flags" --extra-ldflags="-L$prefix/lib" "${hw[@]}")
case "$family" in
    linux)
        ff_args+=(--enable-cross-compile --arch="$arch" --target-os=linux --cc="$CC" --cxx="$CXX" --ar="$AR" --ranlib="$RANLIB" --nm="$NM"
            --strip="$STRIP" --extra-ldsoflags="-s" --extra-libs="$cxx_runtime")
        ;;
    windows)
        ff_args+=(--enable-cross-compile --arch="$arch" --target-os=mingw32 --cross-prefix="$mingw/$cross" --cc="$CC" --cxx="$CXX"
            --extra-ldsoflags="-s" --extra-libs="$cxx_runtime")
        ;;
    darwin)
        ff_args+=(--arch=arm64 --target-os=darwin --cc="$CC" --cxx="$CXX" --install-name-dir=@rpath
            --extra-ldflags="-L$prefix/lib -arch arm64 -mmacosx-version-min=11.0 -Wl,-rpath,@loader_path" --extra-libs="$cxx_runtime")
        ;;
esac
if [ -n "$nasm" ]; then
    ff_args+=(--x86asmexe="$nasm")
fi

rm -rf "$build/ffmpeg"
mkdir -p "$build/ffmpeg"
(
    cd "$build/ffmpeg"
    if ! run "$(source_dir ffmpeg)/configure" "${ff_args[@]}" > "$build/ffmpeg.configure.log" 2>&1; then
        tail -40 "$build/ffmpeg/ffbuild/config.log" >&2
        exit 1
    fi
    if [ "$family" = linux ]; then
        # The glibc 2.28 stubs still export sysctl(), but zig's (current) glibc headers no longer have sys/sysctl.h.
        sed -i.bak 's/^#define HAVE_SYSCTL 1$/#define HAVE_SYSCTL 0/' config.h
        rm -f config.h.bak
    fi

    run make -j"$jobs" > "$build/ffmpeg.build.log" 2>&1 || { grep -E "error" "$build/ffmpeg.build.log" | head -20 >&2; exit 1; }
    run make install > "$build/ffmpeg.install.log" 2>&1
)
cp "$build/ffmpeg/ffbuild/config.log" "$build/ffmpeg.config.log"

# ---------------------------------------------------------------------------------------------------------------------------------
# Output: one real file per library under the name the loader opens (no symlinks; NuGet and zip packages do not keep them).
# ---------------------------------------------------------------------------------------------------------------------------------
rm -f "$out"/*
libraries=(avutil swresample swscale avcodec avformat avfilter)
for library in "${libraries[@]}"; do
    major="$(grep -E "define LIB$(echo "$library" | tr a-z A-Z)_VERSION_MAJOR " "$(source_dir ffmpeg)"/lib$library/version*.h | awk '{print $3}')"
    case "$family" in
        linux) cp -L "$prefix/lib/lib$library.so.$major" "$out/lib$library.so.$major" ;;
        windows) cp "$prefix/bin/$library-$major.dll" "$out/" ;;
        darwin)
            cp "$prefix/lib/lib$library.$major.dylib" "$out/"
            "$STRIP" -x "$out/lib$library.$major.dylib"
            ;;
    esac
done

# What was enabled, for the notices and the smoke test.
{
    echo "FFmpeg $(basename "$(source_dir ffmpeg)") for $rid"
    grep -E "^(License|External libraries|Enabled (decoders|encoders|muxers|demuxers|hwaccels))" -A 40 "$build/ffmpeg.configure.log" || true
} > "$out/ffmpeg-configuration.txt"
grep -oE "^License: .*" "$build/ffmpeg.configure.log" >&2 || true
if ! grep -qE "^License: LGPL version 2.1 or later" "$build/ffmpeg.configure.log"; then
    echo "FFmpeg is not configured as LGPL 2.1 or later; refusing to ship it" >&2
    exit 1
fi

echo "built $out:"
ls -l "$out"
if [ -z "${SLF_KEEP:-}" ]; then
    rm -rf "$build"
fi
