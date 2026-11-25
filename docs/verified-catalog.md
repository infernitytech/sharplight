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
      "kind": "image-generation",     // Sharplight.Models.ModelKinds, "embeddings" or "reranker"
      "family": "FLUX.1",
      "architecture": "flux",
      "level": "verified",            // verified | partial | loads | verified-tiny (only verified and partial are installable)
      "license": "apache-2.0",        // from the model card of the variant's repository
      "gated": true,                  // a repository needs its terms accepted and a token
      "publisher": { "name": "Black Forest Labs", "countryOfOrigin": "DE", "url": "https://bfl.ai" },
                                      // who made the model (not who repackaged it); ISO 3166-1 alpha-2; used by the model feed
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
          "companions": ["clip-l", "t5xxl", "vae"],
          "quality": "near-lossless",                // optional: tier relative to referencePrecision (see "Variant ladders")
          "qualityNote": "Practically identical to BF16.",
          "level": "partial",                        // optional: lower than the entry's level for this variant only
          "verification": "Not compared with the reference on this file; ..."   // required with a partial level on a verified entry
        }
      ],
      "companions": [
        { "id": "t5xxl", "role": "text-encoder-2", "label": "T5-XXL encoder (Q8_0)", "repository": "city96/t5-v1_1-xxl-encoder-gguf",
          "revision": "...", "include": ["t5-v1_1-xxl-encoder-Q8_0.gguf"], "primary": "...", "files": [ ... ] }
      ],
      "referencePrecision": "BF16",   // optional: what the variants' quality is relative to (BF16, or the release format: MXFP4)
      "verification": { "reference": "diffusers", "summary": "...", "providers": ["CPU", "Vulkan"], "date": "2026-11-15", "tests": [] },
      "notes": "...",
      "licenseNotice": "...",         // optional: what the user must know about a non-permissive licence (see below)
      "doc": [{ "section": "4", "cells": { "repositories": "..." } }]   // rows in compatible-models.md; cells override derived text
    }
  ]
}
```

### Licence notices

`licenseNotice` is an optional one-to-two-sentence note (at most 600 characters) on the licence: non-commercial use, a custom agreement, a
revenue or user limit, use restrictions, gated terms. `notes` stays for technical limitations and is never shown as a licence notice.
Every entry whose `license` is recorded and not a known permissive id (`apache-2.0`, `mit`, `bsd-*`, `cc-by-4.0`, `cdla-permissive-2.0`...,
see `CatalogLicenses`) must have one; `validate` reports the others. Read the model's licence on its Hub page when you write it. Studio
shows the notice once per model (a dismissible callout remembered in the settings, on the page of the model's kind and in Chat) and on the
Discover page of the entry; for a non-permissive licence without a notice (a catalog from another source) it shows a generic line.
Readers that do not know the field ignore it (the schema does not forbid extra members), so the schema version stays 1.

Formats: `gguf`, `safetensors`, `pytorch` (`.bin` / `.pth`), `diffusers`, `single-file`, `whisper-cpp`, `onnx`, `slm`. Companion roles
are the auxiliary roles of `Sharplight.Vision.AuxiliaryRoles` (`mmproj`, `vocoder`, `text-encoder`, `text-encoder-2`, `text-encoder-3`,
`vae`, `transformer-2`, `image-encoder`, `draft`, `lora`): at install time each companion is passed to the converter in that role, so the
installed package is complete and the Image, Video and Speak pages load it without picking companion files by hand. A companion with
`"intoVariantFolder": true` is instead downloaded into the variant's own folder and not passed in a role: for official releases whose
files are split over repositories while the importer expects them in one folder (SeedVR2 7B: the DiT and VAE of `SeedVR2-7B` with the
text embeddings `pos_emb.pt` / `neg_emb.pt` of `SeedVR2-3B`). Its `role` still names what the files are; hosts that do not know the
field pass them in that role.

Entries without variants are documented only (rows of compatible-models.md for models whose exact files are not known, families, tiny
checkpoints); Studio lists the installable ones (`verified` or `partial` with at least one variant; partial entries carry a badge and
their limitation).

### Variant ladders

An entry may offer several variants of the same model, from small low-bit files to the reference precision; the model feed publishes
each as its own package in one model group (docs/model-feed.md "Model groups and variants"). Per variant:

- `quality`: `reference`, `near-lossless` (8-bit, FP8), `high` (5-6 bit), `good` (4 bit), `reduced` (3 bit) or `low` (1-2 bit), relative
  to the entry's `referencePrecision`; `qualityNote` says it in one sentence.
- `level` / `verification`: the entry's `level` and `verification` describe what was compared with the reference. A variant whose own
  files were not compared (only the architecture or another variant of the model was) is marked `"level": "partial"` with a
  `verification` note saying what was and was not checked; it stays installable, Studio and the feed show it as partial. A variant
  level is never above the entry's.
- Only formats the engine imports (docs/model-support.md "Weight formats") are offered; `recommended` marks the variant offered first
  (one per entry), normally the 4-bit one when it is verified.

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
(SNAC), OuteTTS (WavTokenizer) and F5-TTS (Vocos), plus Kokoro, whisper.cpp, Silero VAD and a GGUF without companions; on 2026-11-17 also
FLUX.2 Klein 4B Q8_0 and Qwen-Image 2.1 / 2.1 Turbo Q8_0 (pipeline text encoder and VAE), Wan 2.2 TI2V 5B Q8_0 and HunyuanVideo Q4_K_M (the
pipeline folder without its transformer weights, one download in the text-encoder and VAE roles). Every one built a complete package.
The Wan 2.2 A14B and Wan 2.1 I2V / FLF2V installs use the same folder subset (Wan 2.1's) and the roles of their verification tests; their
conversion was not run through the installer (the Wan 2.1 umT5 files were not available locally). `RealCatalogInstallTests` installs whisper-tiny from the committed catalog against the real Hub
(`SHARPLIGHT_HUB_NETWORK_TESTS=1`).

### Rules (CatalogValidator, run by the tests)

- schema version 1, `updated` and verification dates are `yyyy-MM-dd`;
- entry ids unique and well formed, kinds known, at most one recommended variant, doc rows name a defined section;
- installable entries have a verification summary, providers and date;
- every variant and companion has a repository id, a 40-hex lower-case revision, a non-empty file list (unique relative paths, sizes,
  a sha256 or git oid each) and a primary path that is a listed file or a directory of one;
- formats and roles known, companion ids unique, every companion used by a variant and every variant's companions defined; a companion
  that goes into the variant's folder has no file of the same path as the variant's;
- a variant `level` is verified or partial and not above the entry's; a partial variant of a verified entry has a `verification` note;
  `quality` is a known tier.
- (catalog tool `validate` only) `updated` is not older than the latest verification date: set it to the day of every change, since
  hosts replace their copy only with a catalog of a later (or the same) date. Entries verified on 2026-11-24 went into the catalog of
  2026-11-20 without a new date, and Studio builds of 2026-11-20 to 2026-11-23 kept their embedded copy without them.

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

The catalog is also the source of the model feed (docs/model-feed.md): `tools/model-feed` builds one signed, preconverted package per
installable variant at its pinned revision, with license, publisher and verification taken from the entry.

Studio embeds the catalog at build time and, when online, fetches `https://download.sharplight.dev/catalog/verified-models.json`
(at most once a day), uses it when its schema version is supported, it validates and its `updated` date is not older than the
embedded copy's (on the same date the published copy wins), and caches it in Studio's data directory; offline, or when the download is not valid, the cached or embedded copy is used.
The `catalog` job of `.github/workflows/studio.yml` validates the catalog (and that compatible-models.md matches it) on every run and,
on pushes to develop, uploads it to R2 as `catalog/verified-models.json` (bucket and credentials of the `studio-release` environment, the
same as the Studio packages; `STUDIO_CATALOG_KEY` overrides the key), so Studio installs pick up newly verified models without a
release. CI (`ci.yml`, step "Verified catalog") runs `validate`, `docs --check` and `schema --check`, so the catalog and the document
cannot drift apart.
