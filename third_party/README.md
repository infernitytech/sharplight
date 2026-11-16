# Third-party sources

Vendored, unmodified upstream sources that Sharplight compiles into its own native libraries
(`native/sharplight_audio`, `native/sharplight_video`). Nothing in this directory is our code; it is only ever changed by `update.sh`, which re-fetches the
pinned archive from `sources.json`, verifies its SHA-256 and replaces the folder. Build-system files, tests and documentation
of the upstream projects are not vendored (see `prune` in `sources.json`); the source files themselves are untouched.
Only permissive licenses are accepted (no GPL / LGPL).

| Component | Upstream | Version | SHA-256 of the source archive | License | Folder |
| --- | --- | --- | --- | --- | --- |
| miniaudio (device I/O; WAV, FLAC, MP3 decoding; includes dr_wav, dr_flac, dr_mp3; `extras/decoders/libopus` and `extras/decoders/libvorbis`, the backends that plug Opus and Vorbis into its decoder) | https://github.com/mackron/miniaudio | 0.11.26 | `838ba55a21c9142adf7b76bdc4351e5cd8b7ae128f46b25ec25f94ae9969c600` | Public domain (Unlicense) or MIT-0 | `miniaudio/` |
| libopus (Opus codec, float build, portable C) | https://github.com/xiph/opus | 1.6.1 (`v1.6.1`) | `bf0b97ec7a65890b8db90ef94c4d6c18de12584c3085031953a10986f5917745` | BSD-3-Clause | `opus/` |
| libogg (Ogg container) | https://github.com/xiph/ogg | 1.3.6 (`v1.3.6`) | `95b643da661155d79db9de2fca55daed3a8d491039829def246aacb3d9201c81` | BSD-3-Clause | `ogg/` |
| opusfile (Ogg Opus decoding; HTTP support disabled) | https://github.com/xiph/opusfile | 0.12 (`v0.12`) | `a20a1dff1cdf0719d1e995112915e9966debf1470ee26bb31b2f510ccf00ef40` | BSD-3-Clause | `opusfile/` |
| libvorbis (Ogg Vorbis decoding; the encoder part is not used) | https://github.com/xiph/vorbis | 1.3.7 + master commit `1b75110b5a2754ba1931d82dd83cb822b266a21d` (2026-08-04) | `bd0f08c8a63debdce27dd71de7d953f9c92e410d1cfe263220bcb49fff2ab749` | BSD-3-Clause | `vorbis/` |
| libvpx (VP8 and VP9 encoders; generic C configuration, no assembly; decoders, SIMD and postprocessing pruned. The generated build headers are in `native/sharplight_video/config/`) | https://github.com/webmproject/libvpx | 1.15.2 (`v1.15.2`) | `26fcd3db88045dee380e581862a6ef106f49b74b6396ee95c2993a260b4636aa` | BSD-3-Clause + patent grant (`PATENTS`) | `libvpx/` |

Vorbis decoding uses libvorbis, not stb_vorbis: stb_vorbis crashes on malformed headers (it frees uninitialized comment entries after a failed allocation), libvorbis is the fuzzed reference decoder. The release 1.3.7 (2020) still divides by zero in `res2_inverse` on a crafted stream (found by our fuzz test under UBSan); the fix is in the project's master, so that commit is pinned until a release follows. FLAC is decode-only (dr_flac inside
miniaudio), so libFLAC is not vendored. The Opus DNN features (DRED, deep PLC) are not built.

## Build toolchain (not vendored)

The library is cross-compiled with zig `cc` (pinned in `native/sharplight_audio/install-zig.sh`, SHA-256 verified); zig itself is a build
tool and nothing of it ships in the packages.
