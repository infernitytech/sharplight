# Model format v2: portable graphs with tiered, optional precompiled artifacts

Status: container v3 (v2 plus control-flow subgraphs and data-dependent symbols in the IR), artifact tiers, inspection API,
device profiles, `sharplight-convert` (convert, info, profile, precompile --device/--profile, strip, compact), the graph
optimization phase (run by `convert` unless `--no-optimize`) and the package viewer (`Sharplight.PackageViewer`) are
implemented; a separate `optimize` command is not yet (2026-11-13).

## Goals

- One `.slm` file that runs anywhere: device-independent IR graphs and weights are the only required content.
- The stored graphs are already **optimized** (a device-independent optimization phase ran at conversion), so loading does not
  repeat that work.
- **Optional** precompiled artifacts, in any number and for any execution provider, in tiers of decreasing portability:
  - **Code** – intermediate device code that is independent of the driver: SPIR-V for Vulkan (later PTX for a CUDA provider,
    DXIL, Metal IR, ...). Keyed by a *device profile* (provider, code target, feature set, limits), not by a physical device.
  - **Binary** – driver-specific binaries: the Vulkan driver pipeline cache (later CUBIN/SASS, ...). Keyed by provider, device
    identity and driver version.
  A model can ship with no artifacts, with code for some profiles, and/or with binaries for some well-known device/driver pairs.
- Extensible: providers, tiers and artifact kinds are named by strings and described by key/value descriptors; a reader skips
  what it does not understand. Adding a provider needs no format change.
- Tooling: a converter CLI (`sharplight-convert`) and a small Avalonia viewer (`Sharplight.PackageViewer`).

## Graph optimization phase (Sharplight.Core)

A pass pipeline over `ComputeGraph`, independent of devices:

- `IGraphPass` (`Name`, `Run(ComputeGraph) -> ComputeGraph`), `GraphOptimizer` runs an ordered list to a fixed point where a
  pass asks for it.
- Initial passes: dead-node elimination, common-subexpression elimination, IR constant folding (ops whose inputs are all
  `Constant`/weight-free constants, size-capped), reshape/permute chain simplification, identity removal.
- The definition records which pipeline version ran (`ModelDefinition.Properties["graph.optimized"] = "<pipeline id>"`). Loading a
  definition without it (or with an older id) optimizes once at load; a converted package stores the result and loading skips it.
- Importers keep their own source-level folding (e.g. ONNX `OnnxFold`), the IR phase catches what remains and what every importer
  shares.

Implemented in `Sharplight.Graph.Optimization` (pipeline id `sharplight-ir/1`): `GraphOptimizer.Default` runs `IdentityElimination`,
`ViewChainSimplification`, `ConstantFolding` (bit-exact with the CPU backend, results up to 256 KiB), `CommonSubexpressionElimination`
and `DeadNodeElimination`, also inside `If` branches. Values that are graph outputs, nodes that read or write state, own
data-dependent symbols or run subgraphs are never removed or merged; graph inputs are kept as they are.
`ModelAdapterRegistry.Build` and `OnnxGraphAdapter.Build` call `GraphOptimizer.PrepareForLoad`; `AdapterOptions.OptimizeGraphs = false`
or `SHARPLIGHT_NO_GRAPH_OPT=1` skips the phase. A converter calls `GraphOptimizer.Default.Optimize(definition)` and gets per-pass
node counts (`ModelOptimizationResult.Statistics`).

## Container (version 3)

Unchanged header and section table (v1 and v2 files stay readable; any write, including `AddArtifacts` and `Strip`, stores the
current version). Sections:

| kind | name | content |
|---|---|---|
| DEFN | `ir.2` (empty in v1/v2) | IR graphs, states, properties, metadata (optimized) |
| WIDX / WDAT | | weight index / weight bytes, 64 KiB aligned |
| ARTS (5) | artifact-set id | one artifact set (new in v2) |
| KSET (4) | v1 target | v1 kernel set; read as an artifact set of tier `mixed` |

The DEFN section name selects its layout, so a file whose header `AddArtifacts` raised to version 3 keeps reading its older
definition correctly:

- **Layout 1** (empty name, written by v1 and v2): per graph the inputs, the node records (op, group, attributes, operand
  references, output names), outputs and runtime scalars.
- **Layout 2** (`ir.2`, written since v3) adds what runtime control flow and data-dependent shapes need. Each node record is
  followed by its **subgraph attributes** (`i32 n`, then n × (attribute name, nested graph record in the same layout)): the
  `then_branch` / `else_branch` of `If` and the `body` of `Loop` are complete graphs, rebuilt and type-checked like the top-level
  ones (their weights are package weights referenced by name, so `Save` also stores the weights only a subgraph reads). Each
  graph record ends with its **dynamic symbols** (`i32 n`, then n × (name, bound expression text)), declared under their exact
  names on load because node attributes (`count_symbol`, `output_symbols`) refer to them. Packages can therefore store models
  with runtime `If`, `Loop` and `Scan` and with data-dependent ops (`NonZero`, `NonMaxSuppression`, `Unique`, ...).

An **artifact set** is: descriptor (string → string, at least `provider`, `tier`, `kind`), then entries (key → bytes). ARTS
layout: `i32 format (1)`, `i32 n` + n key/value strings (keys sorted), `i32 m` + m × (key, `i64` offset relative to the section,
`i64` length), entries 16-byte aligned. The section name is the set id `provider.tier.kind.<16 hex of SHA-256 over the sorted
pairs>`, so storing a set with an identical descriptor replaces the old one. Descriptors written today:

```
provider=vulkan tier=code   kind=spirv          target=spirv_1_6 subgroup=32 shared=49152 features=bf16,fp16,... compiler=slang-2026.22.0+<build>
provider=vulkan tier=binary kind=pipeline-cache vendor=10de device=<pci device id> cache=<pipelineCacheUUID> driver=988e4100
```

(A future CUDA provider would write e.g. `provider=cuda tier=code kind=ptx sm=90`.) Well-known keys are in
`ArtifactDescriptorKeys`, tiers in `ArtifactTiers` (`code`, `binary`, `mixed` for v1 kernel sets). A v1 KSET is read with the
descriptor `provider=<first field of the target> tier=mixed kind=kernel-set target=<v1 target string>`.

Matching: a device publishes its targets per tier (`IComputeDevice.ArtifactTargets`, a list of `ArtifactTarget`: descriptor,
significant `MatchKeys`, and `KeyPrefixes` that route cache keys to the tier). A stored set matches when provider, tier, kind and
every match key are equal. Vulkan: the code tier matches on `target` only (entries are content-addressed `kernel/<hash>` of
compiler version, source, entry point, target and macros, so a set for a broader or narrower profile is still safe to consult:
a hit is always correct, a miss compiles); the binary tier matches on `vendor`, `device` (model) and `cache` (the driver's pipelineCacheUUID), so one set serves every card of that model on that driver build. Legacy `mixed` sets are
consulted for every tier when their `target` equals the device's v1 `CompilationTarget`. Unknown providers, tiers and kinds are
skipped. The CPU device declares no targets and gets no cache.

Loading (`CompilationArtifacts`, used by `LlmModel`, `EmbeddingModel`, `Reranker`, and by ONNX hosts such as
`sharplight-onnxrun` that compile a package's graph with `CompileOptions.CacheFor = new CompilationArtifacts(source.CompilationCaches).For`;
the load options' `Artifacts` lets several loads share one recording): per device one cache that routes each key to the tier claiming it (the Vulkan
code tier claims `kernel/`, the binary tier is the catch-all for the driver pipeline cache), consults all matching stored sets of
that tier newest first, and records everything compilation used, per tier, for `ExportCompilation(tiers)`.

Appending artifact sets (`ModelPackage.AddArtifacts`) never rewrites weights; superseded sets become dead space, and
`ModelPackage.Compact` / `sharplight-convert compact` rewrites the file without it (the model part is copied byte for byte so
weight offsets stay; live artifact sections are packed after it). `ModelPackage.Strip(path, provider?, tier?)` removes sets by
provider/tier by rewriting only the table.

## Device profiles (offline precompilation)

Lowering depends only on `LoweringOptions` (target, subgroup size, shared memory, features, submission budget, compile options).
`DeviceProfile` (Sharplight.Core, JSON through a source-generated `JsonSerializerContext`) captures what code generation needs,
provider-neutrally: `provider`, `codeTarget` (`spirv_1_6`), `subgroupSize`, `maxSharedMemory`, `features` (provider-neutral
`DeviceFeatures` names: the device's *enabled* features, i.e. detected minus device-level disables), `submissionBudgetMs`, plus
`kind` (it shapes graphs: e.g. no device sampling head on a CPU-class device such as llvmpipe), `totalMemory` (placement; 0 =
unknown, an offline device then assumes 1 TiB), display `name`/`vendor` and informational `properties` (vendor/device ids,
coopmat shapes, ...; never used for compilation).

- Export: `IComputeDevice.CreateProfile()` (null for providers that cannot compile offline); `sharplight-convert profile --device
  vulkan:0 [-o file] [--disable-features ...]`.
- Offline device: providers implementing `IOfflineCompilationProvider` (`CanCompileFor`, `OpenProfile(profile, options)`;
  `ProviderRegistry.OpenProfile` picks one without probing drivers). Vulkan's `VulkanProfileDevice` shares `SpirvCodeGenerator`
  with the live `VulkanDevice` (the one place that turns subgroup size, shared memory, enabled features and budget into
  `LoweringOptions`, lowers, and builds the code-tier `ArtifactTarget`), so the descriptor and every `kernel/<hash>` key are
  identical to a live device of that profile. `Compile` lowers, waits for Slang and returns a segment that cannot run; no
  VkInstance, VkDevice, memory or pipelines. `ArtifactTargets` is only the code tier. `DeviceOptions.DisabledFeatures` and the
  `submission_budget_ms` setting apply as on a live device.
- `sharplight-convert precompile <slm> --profile <file|name>` stores the code tier for the profile (`--tier binary` is rejected:
  binaries need the driver, `precompile --device`).
- Well-known profiles ship embedded in `sharplight-convert` (`profile --list`, `profile --show <name>`), each exported with
  `profile --device` from a real device on the development machine: `rtx-pro-6000-blackwell` (NVIDIA RTX PRO 6000 Blackwell Max-Q,
  driver 610.57.04), `nvidia-no-matrix` (the same export with `--disable-features matrix_f16,matrix_bf16,matrix_int8,matrix_workgroup,bf16`,
  memory left unknown: portable kernels for NVIDIA GPUs with subgroup 32 and 48 KiB shared memory), `radv-raphael` (AMD Raphael
  iGPU, RADV Mesa 26.1.6, subgroup 64) and `llvmpipe` (Mesa 26.1.6 / LLVM 22, subgroup 8). Their kernels were verified to load on
  the respective live device with zero Slang compilations.

Because the code tier matches on `target` only and entries are content addressed, a package may carry code for several profiles;
a device uses whichever entries its own lowering asks for (e.g. `nvidia-no-matrix` code serves the RTX PRO 6000 when it runs with
`--disable-features matrix_f16,matrix_bf16,matrix_int8,matrix_workgroup,bf16`).

## Public inspection API (Sharplight.Package)

`ModelPackageReader.Open(path)` → `PackageInfo` (a snapshot; the file is closed again; no weight or artifact entry bytes are read):
`Version`, `FileSize`, `ReclaimableBytes` (what `Compact` would remove), `Sections` (`PackageSectionInfo`: kind tag, name, offset,
length), `Weights` (`PackageWeightInfo`: name, type, shape, offset, bytes), `ArtifactSets` (`ArtifactSetInfo`: id, descriptor,
provider/tier/kind, legacy flag, entry count, total bytes, `ArtifactEntryInfo` key + size), and `Definition` (`DefinitionInfo`,
decoded on first access: architecture, name, states, raw properties and metadata with `DefinitionInfo.FormatValue` for display,
`GraphInfo` per graph with node count, op histogram, inputs/outputs as `GraphValueInfo` (type, shape, role), weight and state
counts, a lazily built node table of `NodeInfo` (id, op, group, attributes, inputs, outputs), and the lazily built textual IR
`GraphInfo.Ir`: one `IrLine` per line of `ComputeGraph.Dump` (depth, kind, text, scope, node id, defined and used value ids;
control-flow subgraphs get their own scope)). Used by `sharplight-convert info` and the viewer.

## Tools

- `sharplight-convert` (samples/Sharplight.Convert):
  - `convert <src> <dst.slm> [--no-optimize] [--kv-type] [--ctx] [--mmproj] [--shape NAME=DIMS] [--drop-artifacts]`: GGUF,
    Hugging Face directory, ONNX or `.slm` source. Graphs are optimized by `GraphOptimizer` while building (see above). Artifact sets of an `.slm` source are copied (`ModelPackage.ReadArtifactSets`) unless `--drop-artifacts`.
  - `info <slm> [--weights] [--entries] [--nodes <graph>]`.
  - `profile --device <provider:index> [-o file] [--disable-features]`, `profile --list`, `profile --show <name>`.
  - `precompile <slm> (--device <provider:index> | --profile <file|name>) [--tier code|binary|all] [--disable-features] [--ctx]
    [--batch] [--slots] [--kv-tokens] [--bound SYMBOL=VALUE] [--use-kernel-cache]`: graphs are compiled the way their loaders
    compile them, into one shared recording: `LlmModel` (decode, decode_multimodal, vision_encode, and the serving batch decoder
    for `--slots` sequences, default 4 like `sharplight-server`; `--slots 1` skips it), `EmbeddingModel` (embed) and `Reranker`
    (classify) with their default batch limits; other graphs (ONNX) as a `CompiledModel` with the CPU as fallback and the given
    bounds (the ONNX batch symbol defaults to 1). Not compiled: `decode_embeddings` (no loader compiles it; multimodal prompts use
    `decode_multimodal`) and the speculative-decoding variant of the batch decoder (its logits-row bound depends on the
    speculator). By default with a fresh temporary kernel cache; either way the stored pipeline cache holds only this model's
    pipelines (see below). Sets that were served entirely from the package are not stored again.
  - `strip <slm> [--provider] [--tier] [--compact]`, `compact <slm>`.
  - Planned: `optimize <slm>`.
- The driver pipeline cache stored by `precompile --device` and by `sharplight --save` holds only the model's pipelines: the Vulkan
  device remembers which kernels were compiled with each model-level cache; when its own driver cache also holds other
  pipelines (seeded from the per-user cache file, or other models compiled on the device), the export recreates the model's
  pipelines into a fresh driver cache (Qwen3-0.6B: 4.9 MiB instead of the 140+ MiB per-user cache).
- `Sharplight.PackageViewer` (samples/Sharplight.PackageViewer, Avalonia 11; `sharplight-viewer [file.slm]`, open dialog or drag
  and drop): overview (version, size, reclaimable bytes, section table, filterable properties/metadata), graphs (inputs/outputs,
  op histogram, virtualized filterable node table, IR view in the `--dump-graph` format with search, go-to-producer / consumers,
  dataflow neighborhood, linked to the node table), weights (sortable/filterable, totals per type), artifact sets by provider
  and tier with descriptors and entries. Light/dark theme follows the OS. Qwen3-0.6B Q8_0 with both tiers (624 MiB) opens in
  about 0.1-0.15 s; a 100k-node graph builds its node table and IR in about 1.2 s on a background thread.

## Measured load times

Offline code tier (2026-11-13): `precompile --profile rtx.json` (the profile exported from the local RTX PRO 6000) with the
Vulkan loader pointed at a nonexistent ICD (`VK_ICD_FILENAMES=/nonexistent.json`): 133 kernels (decode, 4-slot batch decoder,
embed, classify) in 4.6 s, 4.0 MiB. Loading on the real GPU, cold (fresh `XDG_DATA_HOME`, `__GL_SHADER_DISK_CACHE=0`,
`SHARPLIGHT_LOAD_TIMING=1`): `slang Compiled = 0, DiskHits = 0` for the 107 decode kernels, `Compiled 'decode'` 708-748 ms and
1.2-1.3 s wall, versus 107 Slang compilations, 2.9-3.0 s and 3.5-3.6 s wall without artifacts. With `nvidia-no-matrix` code only:
`--disable-features matrix_f16,matrix_bf16,matrix_int8,matrix_workgroup,bf16` loads its 100 kernels with 0 compilations; without
the disables 48 of 107 kernels are compiled (the matrix-unit ones). ONNX convnext-tiny (`sharplight-onnxrun --provider vulkan`,
cold): `.onnx` 5.6 s compile / 6.2 s wall, precompiled `.slm` (code + binary) 135-194 ms compile / 0.5-0.7 s wall, 115 of 115
kernels from the package.

Qwen3-0.6B Q8_0, RTX PRO 6000 (shared GPU, noisy), `sharplight <model> -p hi -n 1`, cold = empty kernel cache (fresh
`XDG_DATA_HOME`) and NVIDIA shader disk cache off (`__GL_SHADER_DISK_CACHE=0`), 3 runs each; "compile" is `Compiled 'decode'`.

| Variant | Package size | Cold wall (s) | Cold compile (ms) |
|---|---:|---:|---:|
| GGUF | 0.63 GB | 4.8-6.1 | 3900-4900 |
| `.slm`, no artifacts | 615.8 MiB | 6.0-8.5 | 4900-6800 |
| `.slm` + code tier (107 SPIR-V kernels, 3.0 MiB) | 618.8 MiB | 2.8-3.6 | 1400-2200 |
| `.slm` + binary tier (pipeline cache, 3.7 MiB) | 619.5 MiB | 3.8-6.6 | 3000-5200 |
| `.slm` + both | 622.5 MiB | 1.34-1.38 | 390-420 |

Warm (user kernel cache and driver cache populated): GGUF 1.1-1.7 s (compile 560-1000 ms), `.slm` without artifacts 1.1-1.3 s,
`.slm` with both tiers 0.87-1.02 s (compile 260-320 ms).
