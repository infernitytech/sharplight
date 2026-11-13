# Pulling models from Hugging Face

`sharplight-server` can search the Hugging Face Hub, download a model, convert it to a `.slm` package and run or serve it, in
the style of `ollama pull` / `ollama run`. The reusable parts live in two libraries:

| Assembly | Role | Main entry points |
|---|---|---|
| `Sharplight.Packaging` | Convert a source model (any registered importer) to a `.slm` package and precompile kernels into it; shared by `sharplight-convert` and `sharplight-server` | `ModelConverter`, `ConversionOptions`, `PackagePrecompiler`, `PrecompileOptions` |
| `Sharplight.Hub` | Hugging Face client, resumable verified downloads, file selection, local model store, pull pipeline | `HubClient`, `FileDownloader`, `ModelFileSelector`, `ModelStore`, `ModelPuller`, `ModelReference` |

Neither references a format importer: `Sharplight.Hub` only looks at file names (to pick a GGUF quantization or the
safetensors/ONNX files of a checkpoint) and hands the downloaded files to `ModelConverter`, which goes through the
importers the host registered (`GgufImporter.Register()`, `HuggingFaceImporter.Register()`, `OnnxImporter.Register()`).

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
```

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
   precompile kernels (code and binary tiers) for the local GPU (`--no-precompile` skips; `--cpu`, `--device` pick the
   target). A precompile failure is reported but does not fail the pull.
5. Register: rename to `model.slm` and write `manifest.json`. A model without a manifest is invisible, so an interrupted
   pull never leaves a half-registered model; running the same pull again resumes it.
6. Delete the downloaded files (`--keep-source` keeps them under `source/`).

Pulling a model that is already stored at the same commit does nothing (`--force` rebuilds it).

## Store

The store root is, in order of precedence, `--models-dir DIR`, the `SHARPLIGHT_MODELS` environment variable, or `models`
next to the executable. Each model lives in `<root>/<owner>/<repo>/<tag>/` with:

| File | Content |
|---|---|
| `model.slm` | The package |
| `manifest.json` | Repository, tag, alias, revision and commit sha, format, model name and architecture, package size, source files with sizes and SHA-256, whether they were kept, precompiled artifact sets, pull time |
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
