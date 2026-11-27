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
  TimesFM (#89): `google/timesfm-2.0-500m-pytorch`, `google/timesfm-2.5-200m-pytorch` and `-transformers` in the same folder, references from
  `reference_timesfm.py` (venv with torch CPU, transformers >= 5 and the timesfm package from GitHub; fedora-blackwell has it in `~/venvs/timeseries`).
  Moirai (#90): `Salesforce/moirai-2.0-R-small`, `moirai-1.1-R-{small,base,large}` (config.json + model.safetensors), references from
  `reference_moirai.py` (uni2ts installed `--no-deps` from GitHub plus its imports, jax included for jaxtyping's PyTree; fedora-blackwell: `~/venvs/uni2ts`).
  Moirai 1.1 samples a mixture: parity is on the distribution parameters, sampled quantiles are compared in probability space.
  #108: `reference_moirai.py --auto <moirai-1.x>` writes `refs/<name>-auto.json` (uni2ts' `patch_size="auto"` validation losses); Moirai-MoE
  (`moirai-moe-1.0-R-{small,base}`, arch `moirai_moe`, detected by `encoder.centroid` in the safetensors header) references take ~15 min for the
  4000 torch sample paths (`MOIRAI_MOE_SAMPLES=20` for a quick parameter-only run). Spans inputs are (first row, row count), not (start, end).
  Chronos-2 (#91): `amazon/chronos-2`, references from `reference_chronos2.py` (chronos-forecasting >= 2.0; fedora-blackwell: `~/venvs/chronos2`);
  multivariate requests and covariates go through `ITimeSeriesForecaster.ForecastAsync(TimeSeriesForecastRequest)` (default implementation for univariate models).
  HTTP (#92): `POST /v1/timeseries/forecast` (`TimeSeriesApi`, Sharplight.Serving.Tests `TimeSeriesApiTests` / `TimeSeriesRealTests`); independent series go through
  `ForecastBatchAsync` — Chronos-Bolt's graph has a symbolic series axis `S` (old `.slm` files keep the rank-2 graph, one series per pass).
  Chronos-T5 masks missing values (#93) through the encode graph's `key_bias` input (`encoder.key_bias`) and gives the cross-attention only the observed
  encoder rows; the fused `Attention` op's f32 mask combines with `position_bias` / ALiBi on GPU (kernels `main` / `fa`), not with causal or a window.
- LLM architecture bring-up: tiny remote-code checkpoints come from `tests/Sharplight.Families.Tests/tools/mk_tiny_remote_code.py`
  (references: `hf_greedy_reference.py`); architectures llama.cpp added after the 8ca5a80 pin go in `LlamaCppAdditionsFamilyCases.cs`
  and docs/model-support.md section 26. A CPU-only build of llama.cpp HEAD (`cmake -B build -DGGML_CUDA=OFF`, target
  `llama-completion`) takes a few minutes and is the reference for architectures the pin lacks; fedora-blackwell has one in
  `~/scratch/llama.cpp-up` (#22). `llama-completion` refuses GGUFs whose tokenizer adds EOS (ZAYA1: `GGML_ASSERT(!add_eos)`, and a prompt
  ending in an EOG token stops at once); `llama-simple -m X -n N "prompt"` tokenizes like transformers (BOS and EOS added) and is the reference
  there, `--override-kv tokenizer.ggml.add_eos_token=bool:false` makes `llama-completion` usable for chat prompts. ZAYA1 tiny models come from
  `tools/mk_tiny_zaya1.py` (#82; transformers resets the skip expert's class bias to -1 on load). Raon-1 mini tiny models come from
  `tools/mk_tiny_raon1.py` (#84; copies KRAFTON's remote code from `~/models/hf/Raon-1-mini`); llama.cpp b12798 refuses KRAFTON's official
  Raon-1-mini GGUFs (reasoning-effort steering tensors `ffn_effort_steer` it doesn't know), so compare on a copy without those tensors.
- Vision towers (mmproj): `ClipGgufMapping` maps each `clip.projector_type` onto the canonical `VisionParameters` keys that
  `VisionTowerGraph` / `VisionPreprocessor` / `Qwen35Multimodal` interpret; a new projector is usually a table entry plus a few keys.
  Tiny VL checkpoints are compared with transformers in `TinyQwenVlImageTests` (references written by the tools scripts named in
  docs/model-support.md 8.3; the GGUFs come from llama.cpp HEAD's converter, which stores some projector weights in f16 regardless of
  `--outtype`, so round them in the reference too — but check the converted file's tensor types first: LFM2-VL's projector stays f32).
  Bidirectional attention over an image's rows (Gemma 3, mmproj key `image_attention`; #98) goes through CachedAttention's `lookahead` input
  (`InputRole.AttentionLookahead`, symbol `A`: 1 = no row looks ahead, plain kernels; T = the pass has image rows, per-(head, query) kernel);
  every caller of a graph with that input must bind `A`. Towers whose views differ in size
  (LFM2-VL tiles + thumbnail) go through `VisionPreprocessor.PatchifyViews`; the `hybrid` adapter (LFM2) takes a tower like `decoder`.
  `siglip_vl_reference.py` for LFM2-VL needs torchvision in the venv (`uv pip install torchvision` matching the torch build).
  LFM2-VL safetensors (`lfm2_vl`, #110) import into the same canonical tower as the mmproj; a learned position table resampled with
  dense weights (`bilinear_antialias`) must be imported as f32 (the resample is a MatMul with the table as activation; bf16 fails on the CPU).
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
- Don't bake large tables into kernels as `static const` arrays (macro initializer lists): NVIDIA's Vulkan compiler copies a
  dynamically indexed one into every thread's local memory, and at 16 KB per thread (T5-large's position-bias table) that
  corrupted other models' memory and lost the device (#88). Pass them as `const:` buffers (`NewBuffer(BufferKind.Weight, "const:…", init:)`).
  `SHARPLIGHT_VK_PIPELINE_STATS=1` shows each pipeline's "Local Memory Size" (the NVIDIA driver reports it plus 2^36).
- 16-bit recurrent states (`AdapterOptions.RecurrentStateType`, `--reduced-precision` → bf16): `GatedDeltaRule`, `SelectiveScan` and
  `Wkv6/7` keep f32 math and round the state after every token (`state16.slang` on the GPU, `RecurrentState16` on the CPU; the
  chunked delta rule after every 64-row chunk, #11). Keep that rule in new recurrence kernels: it makes outputs independent of how
  rows are batched, which speculative verification and prefix-cache snapshots rely on.
- Output buffers come from a reused arena: a GPU kernel that leaves part of its output unwritten gives a right first run and drifting later
  runs (#111, rank-3 `Gather` tables; now gathered as `[V, prod(rest)]` rows). Tests for such ops should run two steps with different inputs.
- Sharplight.Vulkan.Tests also runs on the CUDA provider with `SHARPLIGHT_TEST_PROVIDER=cuda` (same lowering, NVRTC).
  Speculative (MTP) output equals plain decoding only while row-invariant compiles reproduce the plain single-row
  arithmetic bit for bit. Driver contraction differs per kernel variant and per compiler (NVIDIA Vulkan, RADV and NVRTC
  each differ; #10, benchmarks §123), so after touching matvec / int-dot sums run `*RowInvarianceTests*` on both providers
  and `VerificationRowInvarianceTests` (Families). Row-invariant sums must not imitate what one driver makes of the plain
  expression: give plain and row-invariant SPIR-V compiles the same explicit `fma` / `precise` form (mmvq.slang
  `MQ_FMA_CHAIN`, #102). The Families test needs `~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-{Q8_0,Q4_K_M}.gguf` (unsloth/Qwen3-0.6B-GGUF).

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
