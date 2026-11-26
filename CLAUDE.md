# Notes for Claude Code

Sharplight is an inference engine for LLMs, ONNX models and other model kinds (speech, image/video generation, vision,
audio, ...) in C# for .NET 11, with Vulkan, CUDA and CPU execution providers. Start with README.md and
docs/api-overview.md; docs/roadmap.md has the goals and open work, docs/model-support.md and docs/onnx-support.md the
model coverage, docs/studio.md the Studio app, docs/releasing.md the release process (Nerdbank.GitVersioning,
`release/vX.Y` branches).

Development runs through the MultiClaude board: one issue per branch `issue-<n>`, agent review, the server merges into
`main`. `docs/handoff/` is the pre-migration handoff (2026-11-15) of the earlier single-machine autonomous loop — its
workflow (memory restore via restore.sh, the `wip/*` queue, "push only when asked") no longer applies, but its
toolchain and model-asset notes are still accurate.

## Build and test

- Requires the .NET 11 SDK (`net11.0`). On fedora-ws the SDK comes from dotnetup; non-interactive shells resolve
  `dotnet` to an SDK 10 install first, so run
  `eval "$(~/.dotnetup/dotnetup env script --shell bash --dotnet --dotnetup)"` (or prepend
  `~/.local/share/dotnetup/dotnet` to PATH and set `DOTNET_ROOT` to it) before building. The eval itself exits
  nonzero (a harmless trailing `hash -d`), so don't chain it with `&&`.
- Build: `dotnet build Sharplight.slnx -c Release`. Keep builds niced on shared machines (`nice -n 10`).
- Tests are xunit v3 executables: `dotnet run --project tests/<Suite> -c Release --no-build` (or `dotnet test`).
- Many suites need model assets under `~/models` (tiny-* test models, audio, whisper.cpp, sd, onnx) and skip without
  them; the layout table is in docs/handoff/README.md, tiny models are regenerable with `tools/make-tiny-*.py` /
  `tools/mk_tiny_*.py` / `tools/model-assets`.
- Sharplight.Video.Tests (BundledFfmpegTests) fails, by design, without the bundled FFmpeg libraries in
  `native/ffmpeg/out/<rid>/` (gitignored, so needed once per checkout). Fastest route for linux-x64:
  `multiclaude artifact get builds/sharplight-native-ffmpeg-9.0.2-linux-x64.tar.gz native/ffmpeg/out -x`
  (built by `native/ffmpeg/build.sh`; rebuild and re-upload when the FFmpeg pins in third_party/sources.json change).
  To build it yourself (~10 min): `ZIG=$(native/sharplight_audio/install-zig.sh ~/.local/share/sharplight/zig | tail -1)
  native/ffmpeg/build.sh linux-x64 --jobs 16` (needs cmake, ninja, a host cc (for nasm), pkg-config, python3-venv).
- GPU tests: `SHARPLIGHT_VK_DEVICES=<i>[,<j>]` restricts the Vulkan provider to those device indices — set it to the
  GPU(s) assigned to your session on multi-GPU machines.

## Style and layout

- docs/code-style.md is binding: one type per file, braces always, no single-line scopes, C# 14 extension blocks,
  file-scoped namespaces, XML docs on public members. The mechanical rules are build errors; before committing run
  `python3 tools/check-one-type-per-file.py && python3 tools/check-single-line-scopes.py` (CI runs them too).
- Layering: the core stays format-agnostic (format names only in `Sharplight.Import.*`); importers and
  architecture/graph generation stay outside `Sharplight.Core`; every model kind is storable in `.slm`
  (docs/model-format.md); task clients follow the one DI pattern (`services.AddSharplight()`, `Add<Client>`).

## Rules

- This repository is mirrored PUBLICLY to github.com/infernitytech/sharplight — no secrets, tokens, personal data or
  internal hostnames in commits.
- The enterprise server lives in the private sibling repository `sharplight-server` (same workspace); it consumes the
  engine as NuGet packages and can build against this checkout with `-p:SharplightSource=local`. Keep engine API
  changes and their server callers in the same issue.
