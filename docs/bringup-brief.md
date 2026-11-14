# Model bring-up brief (shared by all architecture agents)

Read first: docs/code-style.md (enforced as build errors; also run `python3 tools/check-one-type-per-file.py`),
docs/model-support.md (matrix, building blocks, waves, llama.cpp reference outputs in section 8.1) and
docs/api-overview.md.

## Architecture rules
- Format-agnostic core: GGUF names live only in src/Sharplight.Gguf, Hugging Face names only in
  src/Sharplight.HuggingFace. Both map onto Sharplight's own neutral canonical names (ModelSource); adapters in
  src/Sharplight.Architectures only see neutral names and neutral hyper-parameter keys. Do not adopt llama.cpp names.
- Modular: a new architecture family gets its own adapter (or builder) files. Shared building blocks (MoE sub-graph,
  norms, activations, attention variants) live in their own well-named helper types that several adapters call. Do
  not grow one giant adapter with flags for everything.
- New ops: add the schema to Core StandardOps (documented inputs/attributes/outputs), a CPU implementation
  (src/Sharplight.Cpu) and a Vulkan lowering + Slang kernel (src/Sharplight.Shaders). Prefer expressing things with
  existing ops first; add a fused op only when it matters for performance.
- Device features: anything that depends on an optional GPU capability is detected, has a neutral feature name, can
  be disabled (`--disable-features`, SHARPLIGHT_DISABLE_FEATURES) and has a fallback path.
- Serving: decoder graphs must keep working with continuous batching (`MultiSequence.Transform`, see
  tests/Sharplight.Vision.Tests/MultiSequenceTests.cs). New recurrent state kinds need the multi-sequence form (runs
  table, per-slot state, fresh run = zero state), or the model must fall back to the session decoder cleanly.

## Verification (required for every architecture)
- Structure test against Hugging Face transformers: tiny-random checkpoints in ~/models/hf (see Families tests and
  their tools/ for how references are generated; a Python venv with transformers may be needed).
- Real model parity: greedy generation on CPU and Vulkan matches the llama.cpp reference output in
  docs/model-support.md section 8.1 (or a reference you generate with the llama.cpp binaries at
  $LLAMA_ROOT/runtime.llama-b11177-{cuda13,vulkan,cpu}/, LD_LIBRARY_PATH = that dir), plus a
  logits-level check where feasible. Add the case to tests/Sharplight.Families.Tests (skips when the model is missing).
- Performance sanity: `sharplight <model> --bench 512` versus `llama-bench -m <model> -p 512 -n 64` (CUDA and Vulkan
  builds). Note results in your report; large gaps are a finding, not necessarily your job to fix.
- Test suites before finishing: Cpu, Vulkan, Gguf, HuggingFace, Families, Vision (multi-sequence), Serving.
- The GPU (96 GB) is shared with the user's apps and other agents: check `nvidia-smi` before loading big models, run
  big models one at a time, and use CPU offload or skip if VRAM is short. Retry timing-only test failures once.

## Process
- Work in your git worktree; `git merge master` first so you are on the latest master. Merge master again before
  finishing and resolve conflicts (other agents work in parallel on other families).
- Small commits ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never push; never touch master.
- Final report: architectures now supported (with test model, CPU/Vulkan parity result, tok/s vs llama.cpp), new ops /
  building blocks, what is left, and anything the other agents need to know.
