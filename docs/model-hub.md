# Pulling models from Hugging Face

`sharplight-server` and `sharplight` can search the Hugging Face Hub, download a model, convert it to a `.slm` package and
run or serve it, in the style of `ollama pull` / `ollama run`. Both resolve the model store the same way (see [Store](#store)),
so with a shared store a model pulled with one is available to the other. The reusable parts live in two libraries:

| Assembly | Role | Main entry points |
|---|---|---|
| `Sharplight.Package` | Read and write `.slm` packages; convert a source model (any registered importer) to a `.slm` package and precompile kernels into it, on a device or for a device profile; shared by `sharplight-convert`, `sharplight-server` and `sharplight` | `ModelConverter`, `ConversionOptions`, `PackagePrecompiler`, `PrecompileOptions` |
| `Sharplight.Hub` | Hugging Face client, resumable verified downloads, file selection, local model store, pull pipeline | `HubClient`, `FileDownloader`, `ModelFileSelector`, `ModelStore`, `ModelPuller`, `ModelReference` |

Neither references a format importer: `Sharplight.Hub` only looks at file names (to pick a GGUF quantization or the
safetensors/ONNX files of a checkpoint) and hands the downloaded files to `ModelConverter`, which goes through the
formats and architectures registered with the host's container (`services.AddSharplight()`, or `services.AddGgufImporter()`,
`.AddHuggingFaceImporter()`, `.AddOnnxImporter()` and `.AddBuiltinArchitectures()`); `ModelPuller` takes the `ModelConverter` service.

The command-line side is shared too: `samples/Sharplight.Samples.Hub` holds the `pull`, `search`, `list`, `show`, `rm` and `store`
commands, the `--models-dir` and pull options, the console progress display and `ModelResolver`, which turns a model
argument (a file, a stored name or a Hugging Face name, pulled when missing) into a package path. `HubCommands.AddTo` adds
the commands to an executable's root command; `sharplight-server` and `sharplight` both do.

## Commands

```bash
sharplight-server search qwen3 --format gguf --limit 10
sharplight-server pull Qwen/Qwen3-0.6B-GGUF:Q8_0        # GGUF quantization by tag
sharplight-server pull unsloth/Qwen3-4B-GGUF             # default Q4_K_M, else Q8_0, else the next listed fallback
sharplight-server pull Qwen/Qwen3-0.6B                   # safetensors checkpoint
sharplight-server list                                   # alias: ls
sharplight-server show qwen3-0.6b:q8_0
sharplight-server run qwen3-0.6b:q8_0                    # interactive chat; pulls first if needed
sharplight-server run Qwen/Qwen3-0.6B-GGUF:Q8_0 -p "Capital of France?"
sharplight-server serve qwen3-0.6b:q8_0 Qwen/Qwen3-0.6B  # pulls missing models, then serves them
sharplight-server rm qwen3-0.6b:q8_0

sharplight list                                          # the same store and commands in the CLI
sharplight qwen3-0.6b:q8_0 --chat -p "Say hi" -n 32       # stored name
sharplight Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M -p "Hi"  # pulled first when not stored (--no-pull fails instead)
```

In `run`, Ctrl+C stops the current answer (the conversation continues with what was generated); at the prompt it quits.

Model names are `owner/repo[:tag]` (a `hf.co/` or `huggingface.co/` prefix is accepted). The tag selects:

- a GGUF quantization (`Q4_K_M`, `Q8_0`, `IQ4_XS`, `UD-Q4_K_XL`, `BF16`, ...) or a file name. Without a tag the puller
  takes Q4_K_M, then Q8_0, then a list of common fallbacks, then the smallest file. Split GGUFs (`-00001-of-00003`) are
  downloaded completely; a vision projector (`mmproj-*.gguf`, F16 preferred) is included when the repository has one.
- `safetensors` or `onnx` to force that format when a repository has several.

Safetensors checkpoints download the root config/tokenizer/template files, the shards named in
`model.safetensors.index.json` (or all root `.safetensors`) and sentence-transformers pooling configs. ONNX repositories
download the model from the root or `onnx/` directory with its external data files.

Stored models are found by full name (`Qwen/Qwen3-0.6B-GGUF:Q8_0`, case-insensitive), by repository when only one tag is
stored, or by the short alias shown by `list` (`qwen3-0.6b:q8_0`: repository name lower-cased without a `-GGUF`/`-ONNX`
suffix). An ambiguous name lists the candidates. `run` and `serve` also accept file paths, and pull missing models unless
`--no-pull` is given.

## Pull pipeline

1. Resolve the revision (`--revision`, default `main`) to its commit sha and list the files with sizes and checksums
   (`api/models/{repo}/tree/{sha}`). All downloads use the pinned commit, so a push to the repository mid-pull cannot mix
   files from two revisions.
2. Select the files, check free disk space for the download plus the package (about 2x the model size) and take a lock
   file so two pulls of the same model cannot interleave.
3. Download each file from `https://huggingface.co/{repo}/resolve/{sha}/{file}` into `<name>.partial`. An interrupted
   download resumes with an HTTP `Range` request; transient failures (connection drops, timeouts, stalls, 408/429/5xx) are
   retried with exponential backoff and honour `Retry-After`. The finished file is checked against the LFS SHA-256 (or the
   git blob sha1 for small files); a mismatch deletes the partial file and fails.
4. Convert to `model.slm.pending` with graph optimization on, reopen it with `ModelPackageReader` to verify it, and
   precompile kernels (code and binary tiers) for the local GPU with `PackagePrecompiler`, the code path of
   `sharplight-convert precompile` (`--no-precompile` skips; `--cpu`, `--device`, `--ctx`, `--batch` pick the target).
   The decode, multimodal and vision graphs and the serving batch decoder for the server's default 4 slots are compiled,
   using the per-user artifact store (the stored sets still hold only this model's kernels). A precompile failure is reported
   but does not fail the pull. To precompile for other GPUs without them, use
   `sharplight-convert precompile model.slm --profile NAME` on the stored package.
5. Register: rename to `model.slm` and write `manifest.json`. A model without a manifest is invisible, so an interrupted
   pull never leaves a half-registered model; running the same pull again resumes it.
6. Delete the downloaded files (`--keep-source` keeps them under `source/`).

Pulling a model that is already stored at the same commit does nothing (`--force` rebuilds it).

## Store

The store root is, in order of precedence:

1. `--models-dir DIR`;
2. the `SHARPLIGHT_MODELS` environment variable;
3. the location chosen in the settings: Studio's settings page, or for the command-line tools the shared file `store.json` in
   the configuration folder of the storage location (`~/.config/sharplight/store.json` in system mode on Linux), written by
   `store use`;
4. the default: the `models` folder of the storage location (portable, system or custom; see [storage.md](storage.md)).

A location is one of three choices:

| Choice | Directory |
|---|---|
| With the other data (default; `Application` in `store.json`) | `<storage root>/models` in portable and custom mode; `sharplight/models` in the data folder in system mode (`~/.local/share/sharplight/models` on Linux, following `XDG_DATA_HOME`; `%LOCALAPPDATA%\sharplight\models` on Windows; `~/Library/Application Support/sharplight/models` on macOS) |
| System folder (`User`) | the system store above, whatever the storage mode is. Shared by every Sharplight executable of the user. |
| Custom folder | Any directory |

In system mode, the tools of a user share one store wherever they are installed. A portable package (`sharplight.portable` next
to it) keeps its models in `data/models` next to itself. If that folder cannot be written (a system-wide install, a read-only mount,
inside a macOS `.app` bundle), the system folders are used and the source says so ("Default: system storage; the portable folder
... is not writable").

```
sharplight store                    # the store in effect and why
sharplight store use user           # or: default, or a directory
sharplight store migrate [--copy]   # move the models of the previous store into the store in effect
sharplight storage use portable     # move everything (settings, models, caches) to another location, see storage.md
```

Earlier versions stored models in a `data` folder next to the executable, or in the per-user folder, which is now the system
store. The store in effect may have no models while another candidate has some. Candidates are the system store, the portable
store of the package, and `data` or `models` next to the package. If nothing chose the store explicitly, the tools then print a
one-line hint, and Studio offers once to keep using that store or to move or copy the models (`ModelStore.FindPreviousStore`,
`ModelStoreTransfer`). Everything that writes models follows the store root: pulls write their partial downloads
(`source/*.partial`), the pending package and the manifest inside the model's directory; Studio converts local models into
`converted/` in the store by default. `sharplight-convert convert` writes exactly the `dst` it is given.

The compilation cache is not part of the model store. Compiled kernels and driver pipeline caches go into the artifact store in
the cache folder of the storage location (`~/.cache/sharplight/artifacts` in system mode on Linux), which every model and tool
shares (see [model-format.md](model-format.md#the-per-user-artifact-store)).

`ModelStore.Resolve` returns the root with its source (`ModelStoreResolution`); applications with their own settings pass
their choice as `ModelStoreSettings`. `list` prints the store it reads. Each model lives in `<root>/<owner>/<repo>/<tag>/` with:

| File | Content |
|---|---|
| `model.slm` | The package |
| `manifest.json` | Repository, tag, alias, revision and commit sha, format, model name and architecture (the family the Hub reports, e.g. `qwen3`), package size, source files with sizes and SHA-256, whether they were kept, precompiled artifact sets, pull time |
| `source/` | Downloaded files while pulling, or kept with `--keep-source` |

## Authentication and network

Gated and private repositories need a token in `HF_TOKEN` (accept the license on the model page first). The token is sent
only to the Hub endpoint, never to the CDN the downloads redirect to, and is never printed. `HF_ENDPOINT` points the client
at a mirror. Requests use a generic `sharplight/<version>` User-Agent. Errors from the Hub are reported plainly: repository
or file not found, gated without a token, access denied, rate limited.

## Tests

`tests/Sharplight.Hub.Tests` runs the client, downloader, selector, store and puller against an in-process fake Hub
(resume, retries, checksum mismatch, quantization selection, split GGUF, search parsing, manifests). Two tests talk to the
real Hub and pull a 610 MB model; they run only with `SHARPLIGHT_HUB_NETWORK_TESTS=1`.
