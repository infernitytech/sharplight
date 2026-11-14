# sharplight-convert

Creates, inspects and maintains Sharplight model packages (`.slm`, see [docs/model-format.md](../../docs/model-format.md)): a
portable model (IR graphs and weights) plus optional precompiled artifact sets per device, in a code tier (driver-independent
kernels, e.g. SPIR-V) and a binary tier (driver pipeline cache).
Conversion and precompilation live in the `Sharplight.Package.Import` library (`ModelConverter`, `PackagePrecompiler`),
which `sharplight-server pull` uses as well.

```sh
C="dotnet run -c Release --project samples/Sharplight.Convert --"

# GGUF, Hugging Face directory, ONNX or another package -> package without artifacts
$C convert ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf qwen3.slm
$C convert model.onnx model.slm --shape pixel_values=-1x3x224x224

# Add precompiled kernels (portable across drivers) and the driver pipeline cache of this GPU
$C precompile qwen3.slm --device vulkan:0 --tier code
$C precompile qwen3.slm --device vulkan:0 --tier binary

# Kernels for a GPU that is not here: export its profile where it is, or use a shipped one
$C profile --device vulkan:0 -o my-gpu.json
$C profile --list
$C precompile qwen3.slm --profile my-gpu.json
$C precompile qwen3.slm --profile nvidia-no-matrix

# Inspect, remove artifacts, reclaim space
$C info qwen3.slm --entries
$C strip qwen3.slm --provider vulkan --tier binary
$C compact qwen3.slm
```

| Command | Meaning |
| --- | --- |
| `convert <src> <dst>` | Builds the model and writes it. `--kv-type`, `--ctx` and `--mmproj` shape LLM graphs; `--shape` sets ONNX input shapes; `--no-optimize` skips the graph optimization (not available yet). The artifact sets of an `.slm` source are kept unless `--drop-artifacts`. |
| `profile --device provider:index` | Prints (or writes with `-o`) the device profile: what kernel code generation depends on. `--disable-features` leaves features out; `--list` / `--show <name>` show the shipped profiles (`rtx-pro-6000-blackwell`, `nvidia-no-matrix`, `radv-raphael`, `llvmpipe`, each exported from a real device). |
| `info <slm>` | Version, sections, graphs, properties, weights, artifact sets; `--weights`, `--entries`, `--nodes <graph>` list details. |
| `precompile <slm> --device provider:index` | Compiles every graph on the device and appends the artifact sets of `--tier code\|binary\|all`: LLM decode / multimodal / vision graphs, the serving batch decoder (`--slots`, `--kv-tokens`), embed and classify graphs, ONNX graphs (`--bound`). Use the `--ctx`/`--batch`/`--slots`/`--disable-features` the model will run with. Uses a fresh kernel cache unless `--use-kernel-cache`. |
| `precompile <slm> --profile file\|name` | The same without the device: lowers and compiles the code tier (SPIR-V) for a device profile. No driver is loaded. |
| `strip <slm>` | Removes artifact sets, optionally only of `--provider` / `--tier`; `--compact` also reclaims the space. |
| `compact <slm>` | Rewrites the file without replaced or stripped sets. |

Loading a package (`sharplight qwen3.slm`) consults every stored set whose descriptor matches the device and compiles the rest.
