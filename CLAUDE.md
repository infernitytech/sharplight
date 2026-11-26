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

Branches: `main` is the unstable integration branch (`0.X-alpha`; its pushes publish the unstable Studio channel, the
prerelease NuGet feed, the verified catalog and the model feed); `release/vX.Y` carries each stable line. The old
`develop` branch was merged into `main` and retired (#47) — never target it.

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
  `tools/mk_tiny_*.py` / `tools/model-assets`. fedora-ws is provisioned (2026-11-26, issue #15): tiny hf/gguf models
  (including the tokenizer-source dirs the generators copy from — they skip the copy silently when the source dir is
  missing, leaving models without `tokenizer.json`), audio, whisper.cpp, sd (incl. full sd15-diffusers weights) and
  the onnx survey set, plus the small real checkpoints some suites load directly (distil-small.en,
  Qwen3-Embedding-0.6B, ERNIE-4.5-0.3B-PT, Llama-3.2-1B-Instruct); only the optional ~800 GB real-checkpoint parity
  models are absent (their tests skip). The stored HF token on fedora-ws is expired — set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`
  when downloading public models there; `glslc` is at `~/.local/bin/glslc`. `$LLAMA_ROOT` (llama.cpp reference builds
  for the bench/ scripts) is not provisioned; a llama.cpp checkout at the pinned parity commit 8ca5a80 sits unbuilt in
  `~/scratch/llama.cpp`.
- Time-series models (Sharplight.TimeSeries.Tests, #25): Hub downloads `amazon/chronos-bolt-*` / `amazon/chronos-t5-*` in
  `~/models/timeseries/<name>`, references from `tools/timeseries-reference/reference_chronos.py <name>...` (venv with torch CPU +
  chronos-forecasting; writes `~/models/timeseries/refs/<name>.json`). Chronos-T5 parity runs greedy (top-k 1); sampling is seeded, not torch-RNG equal.
- LLM architecture bring-up: tiny remote-code checkpoints come from `tests/Sharplight.Families.Tests/tools/mk_tiny_remote_code.py`
  (references: `hf_greedy_reference.py`); architectures llama.cpp added after the 8ca5a80 pin go in `LlamaCppAdditionsFamilyCases.cs`
  and docs/model-support.md section 26. A CPU-only build of llama.cpp HEAD (`cmake -B build -DGGML_CUDA=OFF`, target
  `llama-completion`) takes a few minutes and is the reference for architectures the pin lacks; fedora-blackwell has one in
  `~/scratch/llama.cpp-up` (#22).
- Vision towers (mmproj): `ClipGgufMapping` maps each `clip.projector_type` onto the canonical `VisionParameters` keys that
  `VisionTowerGraph` / `VisionPreprocessor` / `Qwen35Multimodal` interpret; a new projector is usually a table entry plus a few keys.
  Tiny VL checkpoints are compared with transformers in `TinyQwenVlImageTests` (references written by the tools scripts named in
  docs/model-support.md 8.3; the GGUFs come from llama.cpp HEAD's converter, which stores some projector weights in f16 regardless of
  `--outtype`, so round them in the reference too). The language model does not yet model bidirectional attention over image rows
  (Gemma 3, PaliGemma; #98).
- Sharplight.Video.Tests (BundledFfmpegTests) fails, by design, without the bundled FFmpeg libraries in
  `native/ffmpeg/out/<rid>/` (gitignored, so needed once per checkout). Fastest route for linux-x64:
  `multiclaude artifact get builds/sharplight-native-ffmpeg-9.0.2-linux-x64.tar.gz native/ffmpeg/out -x`
  (built by `native/ffmpeg/build.sh`; rebuild and re-upload when the FFmpeg pins in third_party/sources.json change).
  To build it yourself (~10 min): `ZIG=$(native/sharplight_audio/install-zig.sh ~/.local/share/sharplight/zig | tail -1)
  native/ffmpeg/build.sh linux-x64 --jobs 16` (needs cmake, ninja, a host cc (for nasm), pkg-config, python3-venv).
- CPU tiers: `SHARPLIGHT_CPU_ISA=scalar|v128|avx2|avx512|...` caps the kernels' tier (`CpuIsa`); on these Zen hosts v128
  still has FMA3 and VEX encoding. To emulate a real pre-AVX x86 (unfused mul + add), run with `DOTNET_EnableAVX=0`.
  Run Sharplight.Cpu.Tests under `v128` and `scalar` too when changing CPU kernels.
- CPU quantized matmuls: `DeviceFeatures.IntDot` (8-bit activations) applies only below `MatMulOp.IntDotGemmMinRows` (32)
  rows; prompt-sized matmuls take the f32 GEMM (dequantized weight panels) either way (#75). `SHARPLIGHT_CPU_PROFILE=1`
  prints separate decode (T=1) and prefill (T>1) op tables; `=min` prints per-op minima (in an LLM `--bench` these are decode-step times).
- GPU tests: `SHARPLIGHT_VK_DEVICES=<i>[,<j>]` restricts the Vulkan provider to those device indices — set it to the
  GPU(s) assigned to your session on multi-GPU machines.
- Vulkan `a / b` is only 2.5-ULP accurate (RADV, RTX 4090; the Blackwell driver happens to be exact). Where an exact
  IEEE quotient matters (index/tie math that must match the CPU), prefer computing it on the host and passing it as
  constants (nearest Resize reads `ResizeNearestTable` index arrays, #67: the `div_rn` residual fma step in
  `spatial.slang` was not enough on the RTX 4090). The integrated RADV / llvmpipe devices (`SHARPLIGHT_VK_DEVICES=1`/`2` on fedora-blackwell) are handy
  for reproducing such driver-precision bugs.
- Sharplight.Vulkan.Tests also runs on the CUDA provider with `SHARPLIGHT_TEST_PROVIDER=cuda` (same lowering, NVRTC).
  Speculative (MTP) output equals plain decoding only while row-invariant compiles reproduce the plain single-row
  arithmetic bit for bit. Driver contraction differs per kernel variant and per compiler (NVIDIA Vulkan, RADV and NVRTC
  each differ; #10, benchmarks §123), so after touching matvec / int-dot sums run `*RowInvarianceTests*` on both providers
  and `VerificationRowInvarianceTests` (Families).

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
