# Model format v2: portable graphs with tiered, optional precompiled artifacts

Status: container v2, artifact tiers, inspection API and `sharplight-convert` (convert, info, precompile --device, strip,
compact) are implemented; the graph optimization phase, device profiles (`profile`, `precompile --profile`), the `optimize`
command and the viewer are not yet (2026-11-13).

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

## Container (version 2)

Unchanged header and section table (v1 files stay readable; any write, including `AddArtifacts` and `Strip`, stores version 2).
Sections:

| kind | name | content |
|---|---|---|
| DEFN | | IR graphs, states, properties, metadata (optimized) |
| WIDX / WDAT | | weight index / weight bytes, 64 KiB aligned |
| ARTS (5) | artifact-set id | one artifact set (new in v2) |
| KSET (4) | v1 target | v1 kernel set; read as an artifact set of tier `mixed` |

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

Loading (`CompilationArtifacts`, used by `LlmModel`): per device one cache that routes each key to the tier claiming it (the Vulkan
code tier claims `kernel/`, the binary tier is the catch-all for the driver pipeline cache), consults all matching stored sets of
that tier newest first, and records everything compilation used, per tier, for `ExportCompilation(tiers)`.

Appending artifact sets (`ModelPackage.AddArtifacts`) never rewrites weights; superseded sets become dead space, and
`ModelPackage.Compact` / `sharplight-convert compact` rewrites the file without it (the model part is copied byte for byte so
weight offsets stay; live artifact sections are packed after it). `ModelPackage.Strip(path, provider?, tier?)` removes sets by
provider/tier by rewriting only the table.

## Device profiles (offline precompilation)

Lowering depends only on `LoweringOptions` (target, subgroup size, shared memory, features, compile options). A `DeviceProfile`
captures those, can be exported from a live device (`sharplight-convert profile --device vulkan:0 > rtx-pro-6000.json`), and a
few well-known ones ship with the tool. `precompile --profile x.json` lowers every graph for the profile and compiles the code
tier **without the device**. The binary tier needs the real device and driver (`precompile --device vulkan:0`).

## Public inspection API (Sharplight.Package)

`ModelPackageReader.Open(path)` → `PackageInfo` (a snapshot; the file is closed again; no weight or artifact entry bytes are read):
`Version`, `FileSize`, `ReclaimableBytes` (what `Compact` would remove), `Sections` (`PackageSectionInfo`: kind tag, name, offset,
length), `Weights` (`PackageWeightInfo`: name, type, shape, offset, bytes), `ArtifactSets` (`ArtifactSetInfo`: id, descriptor,
provider/tier/kind, legacy flag, entry count, total bytes, `ArtifactEntryInfo` key + size), and `Definition` (`DefinitionInfo`,
decoded on first access: architecture, name, states, raw properties and metadata with `DefinitionInfo.FormatValue` for display,
`GraphInfo` per graph with node count, op histogram, inputs/outputs as `GraphValueInfo` (type, shape, role), weight and state
counts, and a lazily built node table of `NodeInfo` (id, op, group, attributes, inputs, outputs)). Used by
`sharplight-convert info` and the viewer.

## Tools

- `sharplight-convert` (samples/Sharplight.Convert):
  - `convert <src> <dst.slm> [--no-optimize] [--kv-type] [--ctx] [--mmproj] [--shape NAME=DIMS]`: GGUF, Hugging Face directory,
    ONNX or `.slm` source; writes a package without artifacts. The optimization step is a placeholder hook
    (`GraphOptimization.Run`) until the Core optimizer exists. Artifact sets of an `.slm` source are not copied.
  - `info <slm> [--weights] [--entries] [--nodes <graph>]`.
  - `precompile <slm> --device <provider:index> [--tier code|binary|all] [--ctx] [--batch] [--bound SYMBOL=VALUE]
    [--use-kernel-cache]`: language models are compiled the way `LlmModel` compiles them (decode, decode_multimodal,
    vision_encode), other graphs with the given bounds; by default with a fresh temporary kernel cache so the stored pipeline cache
    holds only this model's pipelines. Sets that were served entirely from the package are not stored again.
  - `strip <slm> [--provider] [--tier] [--compact]`, `compact <slm>`.
  - Planned: `optimize <slm>`, `profile --device`, `precompile --profile <json>`.
- `Sharplight.PackageViewer` (Avalonia, planned): open a `.slm`; tree of sections; graph view (node table with op, shapes,
  attributes; op histogram); weights table; artifact sets grouped by provider and tier with sizes; metadata/properties.

## Measured load times

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
