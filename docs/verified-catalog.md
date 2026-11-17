# Verified-model catalog

`catalog/verified-models.json` lists the models that were checked from download to execution, each pinned to the exact Hub revision and
files that were verified, so they can be installed knowing that they work. It is the single source of truth for:

- Sharplight Studio's Discover page ("Verified" mode): per task, the models with their variants (quantizations), sizes, licenses, the
  verification summary and a one-click install that also installs the companions (text encoders, VAEs, vocoders, vision projectors).
- The model tables of [compatible-models.md](compatible-models.md), which are generated from it.

Searching the Hub finds many models that do not work (unsupported architectures, broken conversions, missing tokenizers); the catalog
only holds what passed a comparison with a reference implementation (or is marked partial).

## File format

JSON, schema version 1. The JSON schema is `catalog/verified-models.schema.json` (generated from the types in
`src/Sharplight.Hub/Catalog`, used by editors for completion); `CatalogValidator` checks everything else (see "Rules").

```jsonc
{
  "$schema": "./verified-models.schema.json",
  "schemaVersion": 1,
  "updated": "2026-11-16",            // a newer copy replaces an older one in Studio
  "docSections": [ ... ],             // the generated tables of compatible-models.md (columns per section)
  "entries": [
    {
      "id": "flux.1-schnell",         // unique; lower case, digits, '.' and '-'
      "name": "FLUX.1-schnell",
      "kind": "image-generation",     // Sharplight.Models.ModelKinds, or "embeddings"
      "family": "FLUX.1",
      "architecture": "flux",
      "level": "verified",            // verified | partial | loads | verified-tiny (only verified and partial are installable)
      "license": "apache-2.0",        // from the model card of the variant's repository
      "gated": true,                  // a repository needs its terms accepted and a token
      "tags": ["vision", "moe", "tools", "reasoning", "reranker"],
      "variants": [
        {
          "id": "q8-0-gguf", "label": "Q8_0 GGUF", "format": "gguf", "quantization": "Q8_0", "recommended": true,
          "repository": "city96/FLUX.1-schnell-gguf",
          "revision": "<40-hex commit>",              // the commit the files were verified at
          "include": ["flux1-schnell-Q8_0.gguf"],    // how the files were chosen (or "tag": "Q8_0" for the selector)
          "primary": "flux1-schnell-Q8_0.gguf",      // what the converter opens ("" = the repository root folder)
          "files": [{ "path": "...", "size": 12693000000, "sha256": "<LFS oid>" }],   // or "gitOid" for files in git
          "memoryBytes": 21000000000,                // rough device memory incl. companions (fit estimate only)
          "companions": ["clip-l", "t5xxl", "vae"]
        }
      ],
      "companions": [
        { "id": "t5xxl", "role": "text-encoder-2", "label": "T5-XXL encoder (Q8_0)", "repository": "city96/t5-v1_1-xxl-encoder-gguf",
          "revision": "...", "include": ["t5-v1_1-xxl-encoder-Q8_0.gguf"], "primary": "...", "files": [ ... ] }
      ],
      "verification": { "reference": "diffusers", "summary": "...", "providers": ["CPU", "Vulkan"], "date": "2026-11-15", "tests": [] },
      "notes": "...",
      "doc": [{ "section": "4", "cells": { "repositories": "..." } }]   // rows in compatible-models.md; cells override derived text
    }
  ]
}
```

Formats: `gguf`, `safetensors`, `pytorch` (`.bin` / `.pth`), `diffusers`, `single-file`, `whisper-cpp`, `onnx`, `slm`. Companion roles
are the auxiliary roles of `Sharplight.Vision.AuxiliaryRoles` (`mmproj`, `vocoder`, `text-encoder`, `text-encoder-2`, `text-encoder-3`,
`vae`, `transformer-2`, `image-encoder`, `draft`, `lora`): at install time each companion is passed to the converter in that role, so the
installed package is complete and the Image, Video and Speak pages load it without picking companion files by hand.

Entries without variants are documented only (rows of compatible-models.md for models whose exact files are not known, families, tiny
checkpoints); Studio lists the installable ones (`verified` or `partial` with at least one variant; partial entries carry a badge and
their limitation).

### Revisions

`revision` is the commit the verification ran on. For most entries it comes from the download metadata of the files that were tested
(`.cache/huggingface/download/*.metadata`, which records the commit and the LFS oid of every downloaded file); entries verified before
that record existed were pinned to the repository's `main` at the time the catalog was created (2026-11-16), one or two days after
their verification. Installs always download exactly the pinned commit, so a later change of the repository never changes what
Studio installs; updating a pin (`resolve --update`) needs a new verification.

### Install check

Installing goes through `ModelPuller.InstallAsync` (Sharplight.Hub): exactly the pinned files of the variant and its companions are
downloaded (checked against the catalog's sizes and sha256), then converted to one `.slm` package with each companion in its role and
opened once before the model is registered (manifest field `catalog` = `entry/variant`). On 2026-11-17 the companion wiring was checked this
way with the real converter on the verified files: FLUX.1-schnell Q8_0 (CLIP-L, T5 Q8_0, VAE), Chroma1-HD (FLUX T5 and VAE), SD3.5 Medium
Q8_0 (CLIP-L, CLIP-G, T5, VAE from the pipeline), Z-Image-Turbo Q8_0 and Qwen-Image Q4_K_M (pipeline text encoder and VAE), Orpheus
(SNAC), OuteTTS (WavTokenizer) and F5-TTS (Vocos), plus Kokoro, whisper.cpp, Silero VAD and a GGUF without companions; every one built a
complete package. `RealCatalogInstallTests` installs whisper-tiny from the committed catalog against the real Hub
(`SHARPLIGHT_HUB_NETWORK_TESTS=1`).

### Rules (CatalogValidator, run by the tests)

- schema version 1, `updated` and verification dates are `yyyy-MM-dd`;
- entry ids unique and well formed, kinds known, at most one recommended variant, doc rows name a defined section;
- installable entries have a verification summary, providers and date;
- every variant and companion has a repository id, a 40-hex lower-case revision, a non-empty file list (unique relative paths, sizes,
  a sha256 or git oid each) and a primary path that is a listed file or a directory of one;
- formats and roles known, companion ids unique, every companion used by a variant and every variant's companions defined.

## Tool

`tools/catalog/Sharplight.CatalogTool` (C#, references Sharplight.Hub):

```sh
dotnet run --project tools/catalog/Sharplight.CatalogTool -- validate        # consistency, no network
dotnet run --project tools/catalog/Sharplight.CatalogTool -- resolve         # new entries: commits, files, sizes, sha256, license, memory
dotnet run --project tools/catalog/Sharplight.CatalogTool -- resolve --refresh [--entry ID]   # again at the pinned commits
dotnet run --project tools/catalog/Sharplight.CatalogTool -- resolve --update [--entry ID]    # re-pin to main (verify first!)
dotnet run --project tools/catalog/Sharplight.CatalogTool -- docs [--check]  # regenerate the tables of docs/compatible-models.md
dotnet run --project tools/catalog/Sharplight.CatalogTool -- schema [--check]
dotnet run --project tools/catalog/Sharplight.CatalogTool -- check           # network: every pinned file still exists, same size and checksum
```

`resolve` reads `HF_TOKEN` for gated repositories and uses the Hub API only (metadata, no downloads except safetensors indexes). Files are
chosen by the set's `include` patterns (`*` within a path segment, `**` across segments, `?`) or, without patterns, by
`ModelFileSelector` with the set's `tag` (the selection `sharplight pull` and Studio's Hub search make).

### Adding a verified model

1. Verify it (docs/bringup-brief.md) and note the repository, the exact files and the commit (the download metadata has it).
2. Add an entry with `kind`, `variants` (`repository`, `revision` if known, `include` or `tag`, `label`, `format`, `quantization`),
   `companions` if it needs them, `verification` and a `doc` row for the right section of compatible-models.md.
3. `resolve`, then `docs`, and run the catalog tests (`sl-test Sharplight.Hub.Tests -class "*VerifiedCatalogTests"`).

## Distribution

Studio embeds the catalog at build time and, when online, fetches `https://download.sharplight.dev/catalog/verified-models.json`
(at most once a day), uses it when its schema version is supported, it validates and its `updated` date is newer than the embedded
copy's, and caches it in Studio's data directory; offline, or when the download is not valid, the cached or embedded copy is used.
The `catalog` job of `.github/workflows/studio.yml` validates the catalog (and that compatible-models.md matches it) on every run and,
on pushes to develop, uploads it to R2 as `catalog/verified-models.json` (bucket and credentials of the `studio-release` environment, the
same as the Studio packages; `STUDIO_CATALOG_KEY` overrides the key), so Studio installs pick up newly verified models without a
release. CI (`ci.yml`, step "Verified catalog") runs `validate`, `docs --check` and `schema --check`, so the catalog and the document
cannot drift apart.
