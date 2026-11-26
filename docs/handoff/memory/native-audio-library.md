---
name: native-audio-library
description: "User decision 2026-11-15: build our own native audio library (miniaudio + Opus/Vorbis/FLAC codecs + flat C shim) for all 5 platforms, vendored as isolated third-party code in the repo; own LibraryImport layer"
metadata:
  type: project
---

User (2026-11-15) rejected depending on the Hexa.NET.MiniAudio wrapper (single maintainer, raw 1:1 API) in favor of:
"build miniaudio and the P/Invoke layer ourselves for all platforms, also compile in opus and other format support",
"keep this isolated as 3rd party code somewhere in the repository; it should support all our platforms".

Design agreed: native lib `sharplight_audio` = miniaudio (devices, WAV/FLAC/MP3) + libopus/libogg/opusfile + Vorbis
(stb_vorbis) + optional libFLAC encoder + a small flat C shim (~15 functions, ring buffer in C so no managed callback on
the audio thread). Third-party sources vendored under `third_party/` with licenses + pinned versions (SHA-256),
built by `zig cc` cross-compilation for win-x64, win-arm64, linux-x64, linux-arm64, osx-arm64 (low glibc baseline).
Managed side: `[LibraryImport]` source-generated, AOT/trim-safe; Core only neutral interfaces (decoder/encoder/devices),
package `Sharplight.Audio` registers via `services.AddSharplightAudio()`; Studio replaces its shell-tool playback/
recording with it. Excluded for licensing: AAC/M4A (round 2: OS decoders Media Foundation / AudioToolbox, optional
ffmpeg on Linux), LAME MP3 encoding; WebM container later (libwebm).

**Why:** robust cross-platform audio I/O and formats without a fragile third-party wrapper.
**How to apply:** keep permissive licenses only (no GPL/LGPL static linking); notices file; fuzz decoders.
Related: [[format-agnostic-core]], [[code-style-rules]].
