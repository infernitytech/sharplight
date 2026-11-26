# Model format v2: portable graphs with tiered, optional precompiled artifacts

Status: container v3 (v2 plus control-flow subgraphs and data-dependent symbols in the IR), artifact tiers, inspection API,
device profiles, the two-stage provider precompilation contract, `sharplight-convert` (convert, info, profile, precompile --all-devices/--device/--profile, strip, compact), the graph
optimization phase (run by `convert` unless `--no-optimize`) and the package inspector in Sharplight Studio are
implemented; a separate `optimize` command is not yet (2026-11-13). Since 2026-11-15 a package can describe any model kind
(DESC: kind and components, ATCH: attachments), and precompilation covers every component graph.

## Goals

- One `.slm` file that runs anywhere: device-independent IR graphs and weights are the only required content.
- The stored graphs are already **optimized** (a device-independent optimization phase ran at conversion), so loading does not
  repeat that work.
- **Optional** precompiled artifacts, in any number and for any execution provider, in tiers of decreasing portability:
  - **Code** – intermediate device code that is independent of the driver: SPIR-V for Vulkan (later PTX for a CUDA provider,
    DXIL, Metal IR, ...). Keyed by a *device profile* (provider, code target, feature set, limits), not by a physical device.
  - **Binary** – driver-specific binaries: Vulkan pipeline binaries (VK_KHR_pipeline_binary) or the driver pipeline cache (later
    CUBIN/SASS, Metal binary archives, ...). Keyed by provider, device identity, driver version and kernel ABI.
  The tiers are the two stages of the providers' precompilation contract (see "Two-stage precompilation" below).
  A model can ship with no artifacts, with code for some profiles, and/or with binaries for some well-known device/driver pairs.
- Extensible: providers, tiers and artifact kinds are named by strings and described by key/value descriptors; a reader skips
  what it does not understand. Adding a provider needs no format change.
- Tooling: a converter CLI (`sharplight-convert`) and the package inspector in Sharplight Studio.

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
`ArchitectureCatalog.Build` (and so `ModelLoader.Build`) and `OnnxGraphAdapter.Build` call `GraphOptimizer.PrepareForLoad`; `AdapterOptions.OptimizeGraphs = false`
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
| DESC (6) | `model.json` | the model description (kind and components) as UTF-8 JSON; optional, see below |
| ATCH (7) | attachment name | one attachment (tokenizer file, scheduler config, filter bank, voices, ...), 64 KiB aligned; optional |
| WLAY (8) | | weight layout facts kernels may rely on: `i32 count`, then per weight its name and `i32` scale row block (`WeightInfo.ScaleRowBlock`: rows of each matrix sharing every block scale, e.g. 128 for checkpoints with 128 x 128 FP8 scale blocks); optional, skipped by readers that predate it |
| WACT (9) | | activation schemes of quantized weights: `i32 count`, then per weight its name and `i32` scheme (`WeightInfo.Activations`: 1 weight-only, the products keep full-precision activations; 2 the format quantizes activations too), as the importer read them from the checkpoint's quantization config; weights without an entry are unspecified; optional, skipped by readers that predate it |

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

## Model description: kind and components (DESC, ATCH)

Every model kind lives in the same container: one DEFN section with all graphs of all components, one weight index and one
weight data section (every weight 64 KiB aligned and memory mapped in place, no per-load conversion), and optional artifact sets.
What the model *is* and how its graphs and weights divide into parts is the **model description** (`Sharplight.Models.ModelDescription`),
stored as JSON in a DESC section named `model.json`. Nothing in it is specific to a model kind: the kind and the component roles are
open strings, configuration is free JSON, and attachments are generic named byte ranges. It is written by
`System.Text.Json` with a source-generated context (camelCase names).

```json
{
  "schemaVersion": 1,
  "kind": "speech-recognition",
  "components": [
    { "role": "encoder", "graphs": ["encode"], "weightPrefixes": ["encoder."] },
    { "role": "decoder", "graphs": ["decode"], "weightPrefixes": ["decoder.", "token_embedding.", "output."],
      "symbolBounds": { "T": 448, "ctx": 448, "E": 1500, "R": 1 } },
    { "role": "tokenizer", "attachments": { "tokenizer": "tokenizer/tokenizer.json" } },
    { "role": "feature-extractor",
      "config": { "type": "log-mel", "sampleRate": 16000, "fftSize": 400, "hopLength": 160, "melBins": 80, "center": true,
                  "padMode": "reflect", "trailingPadding": 480000, "dropLastFrame": true, "logBase": "10", "logFloor": 1e-10,
                  "dynamicRange": 8, "logOffset": 4, "logScale": 0.25 },
      "attachments": { "mel-filters": "feature-extractor/mel-filters" } }
  ],
  "config": { "sampleRate": 16000 }
}
```

(Whisper as `EncoderDecoderAdapter` describes it, docs/model-support.md section 13. The tokenizer is the canonical `TokenizerData` JSON
attachment, which `TaskModel.CreateTokenizer(role)` prefers over metadata keys; a component's `T` and `ctx` bounds replace the load
options' batch and context sizes in `TaskModel.CompileComponent` and in precompilation.)

| field | meaning |
|---|---|
| `schemaVersion` | layout version (1). Additions keep the version and readers ignore fields they do not know; a reader rejects a newer version |
| `kind` | what the model does (`ModelKinds`: `text-generation`, `speech-recognition`, `speech-synthesis`, `image-generation`; open-ended, e.g. `depth-estimation`, `time-series`) |
| `components[].role` | the part (`ModelComponentRoles`: `encoder`, `decoder`, `tokenizer`, `feature-extractor`, `text-encoder`, `denoiser`, `vae`, `scheduler`, `vocoder`; open-ended, numbered variants such as `text-encoder-2`) |
| `components[].graphs` | names of the component's graphs in DEFN; empty for configuration-only components (tokenizer, scheduler) |
| `components[].weightPrefixes` | name prefixes of the weights the component owns (accounting, placement, inspection) |
| `components[].metadataPrefix` | prefix of the component's canonical metadata (e.g. a second tokenizer), or null for the model's unprefixed metadata |
| `components[].config` | component configuration as JSON values; the keys belong to the importer and the task implementation that use them |
| `components[].attachments` | local name → ATCH section name |
| `components[].symbolBounds` | default upper bounds of the component graphs' dynamic symbols (used by precompilation and `TaskModel.CompileComponent`; `T` and `ctx` set the row and context bounds) |
| `config` | model-level configuration |

An image generator, for example, is `kind: image-generation` with `text-encoder` (graph and tokenizer metadata), `denoiser`
(UNet/DiT graph), `vae` (`vae_decode`, `vae_encode` graphs) and a configuration-only `scheduler` component (`predictionType`,
`betaSchedule`, ... in `config`, or its original `scheduler_config.json` as an attachment).

**Attachments** (ATCH sections, `ModelDefinition.Attachments`) carry the files a component needs besides graphs and weights: tokenizer
files, scheduler configurations, mel filter banks, voice embeddings. Each is one section, named by its attachment name (e.g.
`tokenizer/vocab.json`), 64 KiB aligned and read from the mapping without copying (`TaskModel.ReadAttachment`). Importers hand attachments to the adapter through
`ModelSource.Attachments` (canonical names such as `feature-extractor/mel-filters`); the adapter places them in the definition and names
them in the description.

Backward compatibility: packages without a DESC section (every LLM package written before it existed, and packages written today for
models whose adapter sets no description) load unchanged and resolve to a text-generation model with one `decoder` component that
runs all graphs (`ModelDescription.Resolve`). The new sections do not change the container version: older readers skip section kinds
they do not know. `ModelPackage.Save` validates the description against the definition (unique roles, every named graph and
attachment exists); `Compact` keeps DESC and ATCH with the model part.

**Precompilation per component.** For a described model, `PackagePrecompiler` (and `sharplight-convert precompile`) compiles the
graphs of every component, each with the given bounds plus the component's `symbolBounds`, into the same artifact sets. Code-tier
entries are content addressed (`kernel/<hash>`), so the kernels of all components coexist in one set per device profile and a load
finds them whichever component asks; `TaskModel.Compile` / `CompileComponent` consult and record into the package's sets through the
same `CompilationArtifacts` routing as `LlmModel`, and `TaskModel.ExportCompilation` returns what a load compiled.

**Inspection.** `PackageInfo.Attachments` lists the attachments, `DefinitionInfo.Description` gives the (resolved) description and
`DefinitionInfo.HasDescription` tells whether the package stores one. `sharplight-convert info` prints the kind and components; the
Studio package inspector shows them on the overview tab and lists DESC and ATCH in the section view.

## Artifact sets

An **artifact set** is: descriptor (string → string, at least `provider`, `tier`, `kind`), then entries (key → bytes). ARTS
layout: `i32 format (1)`, `i32 n` + n key/value strings (keys sorted), `i32 m` + m × (key, `i64` offset relative to the section,
`i64` length), entries 16-byte aligned. The section name is the set id `provider.tier.kind.<16 hex of SHA-256 over the sorted
pairs>`, so storing a set with an identical descriptor replaces the old one. Descriptors written today:

```
provider=vulkan tier=code   kind=spirv          target=spirv_1_6 subgroup=32 shared=49152 features=bf16,fp16,... compiler=slang-2026.22.0+<build>
provider=vulkan tier=binary kind=pipeline-binary vendor=10de device=<pci device id> device_name=<name> cache=<pipelineCacheUUID> driver=<hex>
                abi=1.256 binary_key=<global VK_KHR_pipeline_binary key>
provider=vulkan tier=binary kind=pipeline-cache  vendor=10de device=<pci device id> device_name=<name> cache=<pipelineCacheUUID> driver=<hex> abi=1.256
```

(A future CUDA provider would write e.g. `provider=cuda tier=code kind=ptx sm=90`.) Well-known keys are in
`ArtifactDescriptorKeys`, tiers in `ArtifactTiers` (`code`, `binary`, `mixed` for v1 kernel sets). A v1 KSET is read with the
descriptor `provider=<first field of the target> tier=mixed kind=kernel-set target=<v1 target string>`.

Matching: a device publishes its targets per tier (`IComputeDevice.ArtifactTargets`, a list of `ArtifactTarget`: descriptor,
significant `MatchKeys`, and `KeyPrefixes` that route cache keys to the tier). A stored set matches when provider, tier, kind and
every match key are equal. Vulkan: the code tier matches on `target` only (entries are content-addressed `kernel/<hash>` of
compiler version, source, entry point, target and macros, so a set for a broader or narrower profile is still safe to consult:
a hit is always correct, a miss compiles); the `pipeline-binary` set matches on `vendor`, `device` (model), `binary_key` (the
driver's global pipeline binary key) and `abi`, the `pipeline-cache` set on `vendor`, `device` and `cache` (pipelineCacheUUID), so
one set serves every card of that model on that driver build. Matching is not trust: each kernel binary entry carries SHA-256
checksums and the pipeline key it was captured under, which the device compares with the key of its own create info before the
driver sees the bytes; a pipeline cache blob's header must name the device and driver build. Anything that does not fit is ignored
(`VulkanDevice.RejectedDeviceCode`) and the kernel is compiled from the portable code. Legacy `mixed` sets are
consulted for every tier when their `target` equals the device's v1 `CompilationTarget`. Unknown providers, tiers and kinds are
skipped. The CPU device declares no targets and gets no cache.

Loading (`CompilationArtifacts`, used by `LlmModel`, `TaskModel` (and its text embedder and reranker), and by ONNX hosts such as
`sharplight-onnxrun` that compile a package's graph with `CompileOptions.CacheFor = new CompilationArtifacts(source.CompilationCaches).For`;
the load options' `Artifacts` lets several loads share one recording): per device one cache that routes each key to the tier claiming it (the Vulkan
code tier claims `kernel/`, the binary tiers `vulkan/pipeline_binary/` and `vulkan/pipeline_cache`), consults all matching stored
sets of that tier newest first, and records the portable code compilation used for `ExportCompilation`. Each kernel uses the best
stage stored for the device automatically: its binary when one matches (no driver compilation), else its portable code (no Slang
compilation; the driver compiles), else it is compiled from source. Device code is not recorded while loading; produce it with
`CompilationArtifacts.CompileDeviceCodeAsync(providers)` (the device stage of each device's provider, as `sharplight --save` does).

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
- Portable stage without the device: `IExecutionProvider.CompilePortableAsync` with the profile (see below; `ExecutionProviderCatalog.TryFindForProfile`
  finds the provider without probing drivers). Vulkan runs the workload on its internal `VulkanProfileDevice`, which shares
  `SpirvCodeGenerator` with the live `VulkanDevice` (the one place that turns subgroup size, shared memory, enabled features and budget
  into `LoweringOptions`, lowers, and builds the code-tier `ArtifactTarget`), so the descriptor and every `kernel/<hash>` key are
  identical to a live device of that profile. It lowers, waits for Slang and returns segments that cannot run; no VkInstance,
  VkDevice, memory or pipelines. `DeviceOptions.DisabledFeatures` and the `submission_budget_ms` setting apply as on a live device.
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

## Two-stage precompilation

Every execution provider implements the same contract on `IExecutionProvider` (Sharplight.Core, no provider names):

| Member | Vulkan | CPU | Metal (planned) |
|---|---|---|---|
| `PortableCodeFormat` | `spirv_1_6` | null | AIR / metallib |
| `CompilePortableAsync(PortableCompilationRequest)` → `PortableCodeSet` (code tier) | SPIR-V for the request's `DeviceProfile`, by running its `CompilationWorkload` on a compile-only device; no driver | empty | AIR from Slang's Metal output, compiled with the Metal toolchain for a profile |
| `CompileDeviceSpecificAsync(device, portable)` → `DeviceCodeSet` (binary tier) | pipeline binaries per kernel (VK_KHR_pipeline_binary) when `kernel_binaries` is enabled, else one dedicated pipeline cache | empty | an `MTLBinaryArchive` with a pipeline per kernel, keyed by GPU family and OS build |

The code sets are opaque keyed entries plus a descriptor (`CodeSet.ToArtifactSet()` stores them as ARTS sections), so the package
format does not change for a new provider; it only declares its `ArtifactTargets` (a code target matched on its portable format
and a binary target matched on device identity, driver and ABI) and maps entries to keys. `kernel_binaries` is a provider-neutral,
disableable `DeviceFeatures` name (detected per device, `DeviceOptions.DisabledFeatures` / `SHARPLIGHT_DISABLE_FEATURES`); without
it, or when the driver cannot capture binaries, the Vulkan device stage falls back to the pipeline cache blob. Neither changes the
generated code, so it is not part of profiles or code descriptors.

`PackagePrecompiler.PrecompileAsync(path, providers | provider+device | provider+profile, options, progress, ct)` drives the
stages for a package: per device it opens the device, takes its profile (`IComputeDevice.CreateProfile`), runs the portable stage
(once per distinct profile: identical GPUs share it) and the device stage (once per device identity and driver), and finally
appends every new set in one crash-safe write (`ModelPackage.AddArtifacts`: sections and table are flushed before the header
switches to the new table; an exception truncates the file back to its original length). It returns a `PackagePrecompileReport`
(per device: portable and device code summaries with sizes and sharing; skipped providers and devices with the reason; stored and
unchanged sets). The compilation workload (`ModelCompilation`) compiles the graphs the way the loaders do (see `precompile` below).

## The per-user artifact store

The artifact sets of a package also have a second location: the per-user `ArtifactStore` (Sharplight.Core). It sits in
`artifacts` in the cache folder of the storage location (`StorageLayout.ArtifactsDirectory`, see [storage.md](storage.md)): that is
`~/.cache/sharplight/artifacts` in system mode on Linux, and `data/cache/artifacts` next to a portable package. A set there
is the same set a package holds: the same descriptor, id (`ArtifactDescriptor.Id`), keys, tier and matching. Each set gets its own
directory named by its id, with the descriptor in `descriptor.json` and one file per entry. An entry file holds the key, the data
and a SHA-256 of the data. Entry files are not ARTS sections, because each one has to be replaced atomically on its own.
Precedence when a device compiles a kernel:

1. matching sets stored with the model (the package's `code` and `binary` tiers),
2. the store: the code-tier set of the device's exact descriptor; device code (`pipeline-binary` and `pipeline-cache` sets of the
   device model and driver) is read by the device itself,
3. compilation; the new portable code is recorded for the model (`ExportCompilation`) and written to the store, and the driver
   pipeline cache is saved to the store's `pipeline-cache` set after each compile.

The package stays the primary location. It is only written when a host precompiles, converts or saves on purpose, and device code
from the store is never exported into a package. The store helps everything a package cannot: GGUF, Hugging Face and ONNX models
loaded from their own files, graphs built in code (tests), and kernel development, where content-addressed keys mean only changed
kernels compile again. A descriptor that does not hash to its directory name, or does not parse, makes its set invisible. An
entry that fails its checksum or names another key is a miss and is deleted. Device code must still pass the same per-entry
validation as device code from a package. A driver update gives new descriptors, so stale device code is never consulted and is
eventually evicted. All writes go to a temporary file that atomically replaces the target, so several processes can share the
store. It is size capped at 2 GiB over all sets: past the cap the least recently used entries are deleted, and descriptors are
kept. `SHARPLIGHT_COMPILATION_CACHE=<dir>` moves the store and `=off` turns it off. `DeviceOptions.KernelCacheDirectory` picks the
directory per device, and `DeviceOptions.UseCompilationCache = false` turns the store off for a device (`sharplight-convert precompile` and
`PackagePrecompiler` compile into a fresh temporary store unless `--use-kernel-cache` / `PrecompileOptions.UseUserKernelCache`
says otherwise; either way the stored sets hold exactly the model's kernels).

## Public inspection API (Sharplight.Package)

`ModelPackageReader.Open(path)` → `PackageInfo` (a snapshot; the file is closed again; no weight or artifact entry bytes are read):
`Version`, `FileSize`, `ReclaimableBytes` (what `Compact` would remove), `Sections` (`PackageSectionInfo`: kind tag, name, offset,
length), `Weights` (`PackageWeightInfo`: name, type, shape, offset, bytes), `ArtifactSets` (`ArtifactSetInfo`: id, descriptor,
provider/tier/kind, legacy flag, entry count, total bytes, `ArtifactEntryInfo` key + size), and `Definition` (`DefinitionInfo`,
decoded on first access: architecture, name, states, raw properties and metadata with `DefinitionInfo.FormatValue` for display,
`GraphInfo` per graph with node count, op histogram, inputs/outputs as `GraphValueInfo` (type, shape, role), weight and state
counts, a lazily built node table of `NodeInfo` (id, op, group, attributes, inputs, outputs), and the lazily built textual IR
`GraphInfo.Ir`: one `IrLine` per line of `ComputeGraph.Dump` (depth, kind, text, scope, node id, defined and used value ids;
control-flow subgraphs get their own scope)). Used by `sharplight-convert info` and the Studio package inspector.

For viewers that show the whole file: `PackageInfo.HeaderLength`, `SectionTableOffset` and `SectionTableLength` locate the
header and the section table (together with `Sections` they cover the file, apart from unused gaps);
`ModelPackageReader.SectionKinds` lists the tags of every section kind this version defines (any other tag is an unknown
kind, shown raw); `ArtifactEntryInfo.Offset` is an entry's absolute file offset; and `NodeInfo.InputValues`
(`NodeInputInfo`: value id, kind, name, type, shape, producing node id or -1) gives the dataflow edges of a graph.

## Tools

- `sharplight-convert` (samples/Sharplight.Convert):
  - `convert <src> <dst.slm> [--no-optimize] [--kv-type] [--ctx] [--mmproj] [--shape NAME=DIMS] [--drop-artifacts]`: GGUF,
    Hugging Face directory, ONNX or `.slm` source. Graphs are optimized by `GraphOptimizer` while building (see above). Artifact sets of an `.slm` source are copied (`ModelPackage.ReadArtifactSets`) unless `--drop-artifacts`.
  - `info <slm> [--weights] [--entries] [--nodes <graph>]`.
  - `profile --device <provider:index> [-o file] [--disable-features]`, `profile --list`, `profile --show <name>`.
  - `precompile <slm> (--all-devices | --device <provider:index> | --profile <file|name>) [--tier code|binary|all] [--disable-features] [--ctx]
    [--batch] [--slots] [--kv-tokens] [--bound SYMBOL=VALUE] [--use-kernel-cache]`: graphs are compiled the way their loaders
    compile them, into one shared recording: `LlmModel` (decode, decode_multimodal, vision_encode, and the serving batch decoder
    for `--slots` sequences, default 4 like `sharplight serve`; `--slots 1` skips it), the text embedder (embed) and reranker
    (classify) with their default batch limits; the graphs of every component of a described model with the component's bounds; other graphs (ONNX) as a `CompiledModel` with the CPU as fallback and the given
    bounds (the ONNX batch symbol defaults to 1). Not compiled: `decode_embeddings` (no loader compiles it; multimodal prompts use
    `decode_multimodal`) and the speculative-decoding variant of the batch decoder (its logits-row bound depends on the
    speculator). By default into a fresh temporary artifact store instead of the per-user one. `--all-devices` precompiles for every device of every available
    provider (as Sharplight Studio does). Sets the package already holds completely are not stored again.
  - `strip <slm> [--provider] [--tier] [--compact]`, `compact <slm>`.
  - Planned: `optimize <slm>`.
- Device code stored by `precompile` and by `sharplight --save` holds only the model's pipelines: the device stage creates the
  pipelines of exactly the portable set's kernels, as kernel binaries or into a fresh driver cache, independent of the per-user
  artifact store.
- The package inspector in Sharplight Studio (*Library* › *Inspect*, or *Open .slm file…*; see [studio.md](studio.md)):
  overview (model, file, source, tokenizer, chat template, states), graphs (a pan/zoom dataflow drawing, the IR view in the
  `--dump-graph` format and a node table, all linked; node attributes, inputs with producers, consumers), weights
  (sortable/filterable, totals per type), artifact sets with the profile they were built for and their entries, metadata, and
  the raw section layout including unknown kinds. It reads no weight or entry data.

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
