# Model format v2: portable graphs with tiered, optional precompiled artifacts

Status: design, being implemented (2026-11-13).

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

## Container (version 2)

Unchanged header and section table (v1 files stay readable). Sections:

| kind | name | content |
|---|---|---|
| DEFN | | IR graphs, states, properties, metadata (optimized) |
| WIDX / WDAT | | weight index / weight bytes, 64 KiB aligned |
| ARTS | artifact-set id | one artifact set (new in v2) |
| KSET | target | v1 kernel set; read as an artifact set of tier `mixed` |

An **artifact set** is: descriptor (string → string, at least `provider`, `tier`, `kind`), then entries (key → bytes). Examples:

```
provider=vulkan tier=code   kind=spirv          target=spirv1.6 subgroup=32 shared=49152 features=bf16,coopmat,int8dot,... compiler=slang-2026.22
provider=vulkan tier=binary kind=pipeline-cache vendor=10de device=<uuid> driver=988e4100
provider=cuda   tier=code   kind=ptx            sm=90 ...
```

Matching: a device publishes its descriptors per tier (`IComputeDevice.ArtifactTargets`); a set matches when every key the
device declares significant for that tier is equal. Code entries are content-addressed (`kernel/<hash>` of source, entry point,
target and macros), so a set for a broader or narrower profile is still safe to consult: a hit is always correct, a miss compiles.

Appending artifact sets (`ModelPackage.AddArtifacts`) never rewrites weights; superseded sets become dead space, and
`sharplight-convert compact` rewrites the file without it. `strip` removes sets by provider/tier.

## Device profiles (offline precompilation)

Lowering depends only on `LoweringOptions` (target, subgroup size, shared memory, features, compile options). A `DeviceProfile`
captures those, can be exported from a live device (`sharplight-convert profile --device vulkan:0 > rtx-pro-6000.json`), and a
few well-known ones ship with the tool. `precompile --profile x.json` lowers every graph for the profile and compiles the code
tier **without the device**. The binary tier needs the real device and driver (`precompile --device vulkan:0`).

## Public inspection API (Sharplight.Package)

`ModelPackageReader.Open(path)` → `PackageInfo`: version, sections (kind, name, offset, length), definition summary (graphs
with node counts, op histogram, inputs/outputs, properties, metadata), weights (name, type, shape, bytes), artifact sets
(descriptor, entry count, total bytes, entry keys and sizes). Used by the viewer and by `sharplight-convert info`.

## Tools

- `sharplight-convert` (samples/Sharplight.Convert): `convert <src> <dst.slm> [--no-optimize]`, `info <slm>`,
  `optimize <slm>`, `profile --device`, `precompile <slm> --profile <json> | --device <id> [--tier code|binary|all]`,
  `strip <slm> [--provider] [--tier]`, `compact <slm>`.
- `Sharplight.PackageViewer` (Avalonia): open a `.slm`; tree of sections; graph view (node table with op, shapes, attributes;
  op histogram); weights table; artifact sets grouped by provider and tier with sizes; metadata/properties.
