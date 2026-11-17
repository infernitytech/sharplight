# Third-party notices

Sharplight.Video itself is MIT licensed. It ships native FFmpeg libraries (`runtimes/<rid>/native/`: `avutil`, `swresample`, `swscale`,
`avcodec`, `avformat`, `avfilter`) that Sharplight builds from unmodified upstream sources with `native/ffmpeg/build.sh` in the Sharplight
repository. The complete license texts are in `licenses/ffmpeg/` next to this file.

## FFmpeg 9.0.2 (https://ffmpeg.org), GNU Lesser General Public License 2.1 or later

FFmpeg is configured LGPL-only: never with `--enable-gpl` or `--enable-nonfree` (the build refuses to finish unless FFmpeg reports
"LGPL version 2.1 or later"), and built as shared libraries that Sharplight loads at run time. You may replace them with your own build of
FFmpeg 9.0.x with the same major library versions (avutil 61, swresample 7, swscale 10, avcodec 63, avformat 63, avfilter 12), for
example by pointing the `SHARPLIGHT_FFMPEG_DIR` environment variable at it.

**Source offer.** The corresponding source code is the FFmpeg 9.0.2 release archive
(https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz, SHA-256 `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`) together
with the build scripts and the pinned dependency list of the Sharplight release that contains these binaries (`native/ffmpeg/` and
`third_party/sources.json` at the matching tag of https://github.com/infernitytech/sharplight); no FFmpeg source file is modified. For at
least three years after each release we also provide a copy of that source on request (open an issue on the repository), at no charge
beyond the cost of the transfer.

The FFmpeg libraries statically include these libraries (all permissive licenses):

| Component | Version | License | Used for |
| --- | --- | --- | --- |
| libvpx (https://chromium.googlesource.com/webm/libvpx) | 1.17.0 | BSD-3-Clause + WebM patent grant | VP8 / VP9 encoding |
| dav1d (https://code.videolan.org/videolan/dav1d) | 1.5.4 | BSD-2-Clause | AV1 decoding |
| SVT-AV1 (https://gitlab.com/AOMediaCodec/SVT-AV1) | 4.3.0 | BSD-3-Clause-Clear + Alliance for Open Media Patent License 1.0 | AV1 encoding |
| OpenH264 (https://github.com/cisco/openh264) | 2.7.0 | BSD-2-Clause | H.264 encoding |
| libopus (https://opus-codec.org) | 1.6.1 | BSD-3-Clause | Opus audio encoding |
| zlib (https://zlib.net) | 1.3.2 | zlib | PNG / APNG, compressed MP4 headers |
| nv-codec-headers (https://github.com/FFmpeg/nv-codec-headers) | 12.2.72.0 | MIT | NVENC / NVDEC (the NVIDIA driver is loaded at run time when present) |
| LLVM libc++ / libunwind / compiler-rt (C++ runtime of OpenH264, from zig on Linux and llvm-mingw on Windows) | | Apache-2.0 WITH LLVM-exception | |
| mingw-w64 runtime (Windows builds, llvm-mingw 20261104) | | mingw-w64 runtime license (public domain / permissive) | |

**Patents.** H.264 and HEVC are covered by patent pools in some countries. OpenH264 is compiled from source here, so Cisco's royalty
coverage (which applies only to Cisco's own prebuilt OpenH264 binaries) does not apply; the hardware encoders (NVENC, VideoToolbox, Media
Foundation) come with the licenses of the GPU or operating system vendor. VP8, VP9 and AV1 come with the royalty-free patent grants
listed above.

## FFmpeg.AutoGen (https://github.com/Ruslan-B/FFmpeg.AutoGen), MIT

The managed bindings (NuGet package `FFmpeg.AutoGen` 9.0.1.1, MIT licensed since version 8) are a package dependency, not embedded.
