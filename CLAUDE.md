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
  `tools/mk_tiny_*.py` / `tools/model-assets`. fedora-ws is provisioned (2026-11-26, issue #15): tiny hf/gguf models,
  audio, whisper.cpp, sd and the onnx survey set are in place; only the optional ~800 GB real-checkpoint parity models
  are absent (their tests skip). The stored HF token on fedora-ws is expired — set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`
  when downloading public models there; `glslc` is at `~/.local/bin/glslc`. `$LLAMA_ROOT` (llama.cpp reference builds
  for the bench/ scripts) is not provisioned; a llama.cpp checkout at the pinned parity commit 8ca5a80 sits unbuilt in
  `~/scratch/llama.cpp`.
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
