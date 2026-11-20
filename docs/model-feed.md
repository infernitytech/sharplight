# Model feed

A model feed is "NuGet for models": preconverted, immutable `.slm` packages with signed manifests, listed in a static, paged and
signed JSON index. A feed is a folder: serve it from object storage, any HTTPS server, or a local or network folder (air-gapped
sites). Hosts (Studio, servers built on Sharplight) install from it without talking to the model hub, converting or needing any
importer. The framework library is `Sharplight.ModelFeed`; the builder is `tools/model-feed` (`feedtool`).

## Relation to the verified catalog

`catalog/verified-models.json` (docs/verified-catalog.md) stays the source of truth for what was verified. The feed is its published,
installable form:

| Verified catalog (source) | Feed (product) |
|---|---|
| upstream files (GGUF, safetensors, ...) at a pinned commit | the converted `.slm`; hosts never convert |
| entry + variant + companions | one self-contained package per variant; companions (projector, encoders, VAE, vocoder) merged into the `.slm` exactly as `ModelPuller.InstallAsync` does |
| `updated` date | a semantic version per package; published versions are immutable |
| no signatures | signed manifests and a signed index |

Only entries with level `verified` or `partial` and at least one variant become packages.

## Package

A package version is three objects in `packages/<id>/<version>/`:

```
<id>.<version>.slm     the model (container v3)
manifest.json          metadata (below)
manifest.sig           detached signature over the exact bytes of manifest.json
```

- **Id**: lower case `[a-z0-9.-]`, `<catalog entry id>.<catalog variant id>` (`qwen3-0.6b.q8-0-gguf`). Ids are never reused.
- **Version**: SemVer 2. `MAJOR`: other weights or behaviour (new upstream revision); `MINOR`: same weights, rebuilt package (newer
  engine, graphs, added portable code, higher minimum engine); `PATCH`: manifest-only change (same `.slm`, new metadata such as kind, tasks, requirements, license or description; the new
  version's manifest names its own copy of the unchanged file, which the feed stores by copying the earlier object, nothing is rebuilt). Prerelease versions are only chosen when
  a host opts in or a range names one.
- The `.slm` is identified by its SHA-256; device-specific code is never part of a package (it is compiled on the host); portable code
  may be.

### manifest.json

```jsonc
{
  "schemaVersion": 1,
  "id": "qwen3-0.6b.q8-0-gguf",
  "version": "1.0.0",
  "display": { "name": "Qwen3 0.6B", "variant": "Q8_0 GGUF", "family": "Qwen3", "summary": "...", "tags": ["reasoning", "tools"] },
  "group": { "id": "qwen3-0.6b", "name": "Qwen3 0.6B", "variant": "q8-0-gguf", "label": "Q8_0 GGUF", "format": "gguf",
             "quantization": "Q8_0", "bitsPerWeight": 8.5, "recommended": true, "quality": "near-lossless",
             "qualityNote": "Practically identical to BF16.", "reference": "BF16", "size": 645685324, "sourceSize": 639446688 },
  "publisher": { "name": "...", "countryOfOrigin": "CN", "url": "..." },   // who made the model; ISO 3166-1 alpha-2
  "packager": { "name": "Sharplight", "feed": "https://.../models/v1" },     // who converted and signed it
  "license": { "spdx": "apache-2.0", "name": "Apache License 2.0", "url": "...", "commercialUse": true, "gated": false },
  "model": {
    "kind": "text-generation", "tasks": ["chat", "completion"], "capabilities": ["tools", "reasoning", "structured-output"],
    "architecture": "qwen3", "parameters": 596049920, "quantization": "Q8_0", "contextLength": 40960,
    "languages": ["en", "zh"], "modalities": { "input": ["text"], "output": ["text"] }, "defaults": { "chatTemplate": "embedded" }
  },
  "requirements": {
    "server": ">=1.0.0",          // minimum host version (version range); optional
    "engine": ">=0.1.0",          // minimum Sharplight engine version
    "packageFormat": 3,           // .slm container version
    "deviceFeatures": [],         // neutral engine feature names a device must have (hard requirement)
    "kernels": [ { "weightType": "q8_0", "bytes": 633233408, "nativeFeatures": ["matrix_f16"], "fallback": true },
                 { "weightType": "f32", "bytes": 262144, "nativeFeatures": [], "fallback": true } ],
    "memory": [ { "device": "gpu", "weights": 633495552, "overhead": 1959878656,
                  "perSlotContext": { "tokens": 4096, "bytes": 469762048 },
                  "perSlotContextMax": { "tokens": 40960, "bytes": 4697620480 },
                  "kvBytesPerToken": 114688, "stateBytesPerSlot": 0 },
                { "device": "cpu", ... } ],
    "disk": 640123456
  },
  "dependencies": [ { "id": "silero-vad-v5.onnx", "range": "[1.0.0,2.0.0)", "kind": "required", "role": "voice-activity" } ],
  "provenance": {
    "catalogEntry": "qwen3-0.6b", "catalogVariant": "q8-0-gguf",
    "sources": [ { "repository": "Qwen/Qwen3-0.6B-GGUF", "revision": "<commit>", "role": "primary",
                   "files": [ { "path": "Qwen3-0.6B-Q8_0.gguf", "size": 639446688, "sha256": "..." } ] } ],
    "converter": { "engine": "0.1.283-alpha", "tool": "feedtool 0.1.283-alpha", "options": { "optimize": "true" } },
    "built": "2026-11-17T10:00:00Z"
  },
  "verification": { "level": "verified", "reference": "llama.cpp", "summary": "...", "providers": ["CPU", "Vulkan"], "date": "2026-11-15",
                    "packageCheck": { "loaded": true } },
  "files": [ { "name": "qwen3-0.6b.q8-0-gguf.1.0.0.slm", "role": "package", "size": 640123456, "sha256": "...",
               "chunkSize": 67108864, "chunks": ["<sha256 of chunk 0>", "..."] } ],
  "state": { "listed": true },
  "notes": "..."
}
```

| Field | Rule |
|---|---|
| `requirements.server`, `.engine`, `.packageFormat` | checked before install and before load (`PackageCompatibility`); an incompatible package is listed but not installable, with the reason |
| `group` | the model this package is a variant of; see "Model groups and variants" |
| `requirements.memory` | first-pass placement filter; the engine's own estimate on the device decides (details below) |
| `requirements.kernels` | the package's weight types and the device features that accelerate them; see below |
| `dependencies` | NuGet-style ranges. `required`: installed together, a missing one fails the install. `optional`: offered. Roles are auxiliary roles (`draft`, `voice-activity`, `lora`, ...) or `pairing`. Companions merged into the `.slm` are not dependencies. |
| `license.gated` | the host asks for explicit acceptance of the terms before install |
| `files[].chunks` | SHA-256 per chunk: resumable, verified transfers from several sources (mirrors, peers) |
| `state` | the state at publication; the current state (deprecated, yanked, unlisted) is in the package index |

Readers ignore unknown fields; `schemaVersion` changes only for breaking changes.

## Feed layout

A feed has channels (the official one: `stable` and `unstable`). Package versions are stored once and shared by every channel; each
channel has its own signed listing:

```
keys/keyset.json, keys/keyset.sig                         feed keys, signed by a root key                        (shared)
packages/<id>/<version>/manifest.json, manifest.sig,
                        <id>.<version>.slm                the immutable package version                         (shared)
<channel>/index.json, <channel>/index.sig                 the channel's service index and its signature
<channel>/catalog/index.json                              list of pages
<channel>/catalog/page/<n>.json                           package summaries, 200 per page, ordered by id
<channel>/catalog/models.json                             the model groups with their variants (see "Model groups and variants")
<channel>/packages/<id>/index.json                        the versions listed in this channel: manifest digest, size, publication
                                                          time, current state in this channel, requirements and dependencies
```

- New versions are published to `unstable`; **promotion** lists an already published version in `stable` as well, without copying
  or re-uploading anything. State (deprecated, yanked, unlisted) is per channel.
- The service index names its channel (`"channel": "stable"`) under the signature, so one channel's index cannot be served as
  another's. Sequence numbers and the rollback state of clients are per channel (`FeedClient.StateKey` = location + `#channel`).
- A host configures the feed root URL plus a channel (default `stable`); switching the channel re-syncs.
- A feed without channels keeps the channel documents at the root (`index.json`, `catalog/`, `packages/<id>/index.json` beside the
  version folders); clients use it with no channel set.

`index.json`:

```jsonc
{
  "schemaVersion": 1,
  "name": "Sharplight verified models",
  "updated": "2026-11-17T10:05:00Z",
  "expires": "2026-12-17T10:05:00Z",     // freeze protection: an expired index is rejected (accepted with a warning when allowed offline)
  "sequence": 412,                        // strictly increasing per channel; lower than last seen is rejected (rollback protection)
  "channel": "stable",                    // the channel this index lists (absent for a feed without channels)
  "resources": { "catalog": "stable/catalog/index.json", "models": "stable/catalog/models.json",
                 "packages": "stable/packages/{id}/index.json", "keys": "keys/keyset.json" },
  "catalog": { "pages": 3, "packages": 412, "digest": "sha256:...", "models": 97, "modelsDigest": "sha256:..." }
}
```

Digests (`sha256:<hex>` over the exact bytes) chain index → catalog index → pages → package index → manifest → `.slm` (and its chunks),
so one signature check on the index authenticates every listing, and the manifest signature authenticates each package on its own:
mirrors and peers need no trust (the model listing hangs off the index the same way: `catalog.modelsDigest`). A page entry carries
what a listing needs without further requests: id, latest version, all versions, display, group, publisher and country of origin,
license, model (kind, tasks, capabilities, size data), requirements, verification (level, reference, summary, providers, date),
dependencies, size, state, manifest and package index digests.

Caching: the channel documents (`<channel>/**`) and the key set short-lived (`no-cache`, ETag); everything under
`packages/<id>/<version>/` immutable (`max-age=31536000, immutable`). Package files are read with range requests. Publication is
atomic from a reader's view: package objects first, package indexes and pages next, `index.json` and `index.sig` last.

## Model groups and variants

A model is usually published in several variants (quantizations and formats): `qwen3.5-122b-a10b.ud-iq2-xxs-gguf`,
`qwen3.5-122b-a10b.ud-q4-k-xl-gguf`, `qwen3.5-122b-a10b.fp8-safetensors`, ... Each variant is its own package (own versions,
own `.slm`); the **model group** ties them together so a host lists the model once and compares its variants against its hardware.
The contract (additive to schema version 1; readers that do not know these fields ignore them):

**Identity.** The group id is the verified-catalog entry id; the variant id is the catalog variant id; the package id is
`<group.id>.<group.variant>`. `feedtool publish` rejects a manifest whose `group` does not name its package id that way, an unknown
`quality` tier, or `bitsPerWeight` outside (0, 64]. A package without `group` (published before groups existed) is grouped by its
`provenance.catalogEntry` / `catalogVariant` when the package id is `<entry>.<variant>` (`PackageManifest.EffectiveGroup`); otherwise it
is a model of its own whose group id is the package id.

**`group`** (manifest, copied into each catalog-page summary):

| Field | Type | Meaning |
|---|---|---|
| `id` | string, required | model group id = catalog entry id (`[a-z0-9.-]`) |
| `name` | string | model name, the same for every variant |
| `variant` | string, required | variant id within the group = catalog variant id |
| `label` | string | what a user picks (`UD-Q4_K_XL GGUF + mmproj`) |
| `format` | string | upstream format the package was converted from: `gguf`, `safetensors`, `onnx`, ... (the `.slm` itself is always the same format) |
| `quantization` | string | upstream quantization name (`Q4_K_M`, `UD-IQ2_XXS`, `FP8`, `MXFP4`, `NVFP4`, `GPTQ-Int4`, `BF16`) |
| `bitsPerWeight` | number | measured on the package: weight bytes x 8 / weight elements, 2 decimals (includes scales and the tensors kept in higher precision) |
| `recommended` | bool | the variant a host offers first; at most one per group in a channel (`feedtool verify` reports more) |
| `quality` | string | tier relative to `reference`, best first: `reference`, `near-lossless` (8-bit, FP8), `high` (5-6 bit), `good` (4 bit), `reduced` (3 bit), `low` (1-2 bit) |
| `qualityNote` | string | one sentence on the quality relative to `reference` |
| `reference` | string | the precision quality is measured against: `BF16`, or the release format (`MXFP4` for gpt-oss) |
| `size` | integer | bytes of the `.slm` (download and disk size) |
| `sourceSize` | integer | bytes of the upstream files the package was built from (with companions) |

The verification level is per variant: `verification.level` of the manifest (and of the summary) is `verified` when this variant's
own files were compared with the reference, `partial` when only the architecture or another variant of the model was (the catalog
variant's `level`; `verification.summary` then starts with what was and was not checked on this variant).

**Model listing** (`<channel>/catalog/models.json`, `FeedModelListing`): written by every publish, promote, state change and reindex
from the same summaries as the pages, chained into the signed index by `catalog.modelsDigest` (and `catalog.models` = number of
models); readers that fetch it check the digest (`FeedClient.GetModelListingAsync`). It lists the listed packages only (unlisted ones
are left out; yanked and deprecated ones stay, with their `state`).

```jsonc
{
  "schemaVersion": 1,
  "models": [                                   // ordered by id
    {
      "id": "qwen3.5-122b-a10b", "name": "Qwen3.5 122B-A10B", "family": "Qwen3.5", "kind": "text-generation",
      "architecture": "qwen3_5_moe", "parameters": 122000000000, "contextLength": 262144,
      "publisher": { ... }, "license": { ... }, "tags": ["moe", "vision", "reasoning", "tools"],
      "recommended": "qwen3.5-122b-a10b.ud-q4-k-xl-gguf",       // package id, or absent
      "variants": [                                             // bitsPerWeight ascending, then size, then id
        { "package": "qwen3.5-122b-a10b.ud-iq2-xxs-gguf", "version": "1.0.0", "variant": "ud-iq2-xxs-gguf",
          "label": "UD-IQ2_XXS GGUF + mmproj", "format": "gguf", "quantization": "UD-IQ2_XXS", "bitsPerWeight": 2.4,
          "recommended": false, "quality": "low", "qualityNote": "...", "reference": "BF16", "size": 37600000000,
          "verification": "partial", "requirements": { ... the package's requirements ... }, "state": { "listed": true } }
      ]
    }
  ]
}
```

Model-level fields (name, family, kind, architecture, parameters, context, publisher, license, tags) come from the recommended
variant, else the first package by id. A feed published before model listings existed has no `resources.models`; then
`FeedSnapshot.Models` (also available on every snapshot) groups the verified page summaries the same way.

**Compatibility check per variant.** Everything a host needs is in `requirements` (pages and the model listing carry it; no manifest
fetch):

- `memory[]`, one entry per device kind (`gpu`: device-local memory of an accelerator; `cpu`: system memory). The two entries carry the
  same figures: the engine keeps one copy of the weights on the device that runs them, and splits layers between devices when one is
  too small (a split needs the sum, plus one overhead per device).
  - `weights`: weight bytes of the package (companions included).
  - `overhead`: activations and scratch for a 512-token prefill batch at the default context plus 384 MiB runtime (at least what the
    catalog's estimate leaves beyond weights and one default context); conservative.
  - `kvBytesPerToken`: KV cache bytes per context token of one slot over every layer whose cache grows with the context, at a 16-bit
    cache (an 8-bit cache, `kv-type q8_0`, needs about 0.53x, `q4_0` 0.28x).
  - `stateBytesPerSlot`: bytes per slot that do not grow with the context: recurrent state (linear-attention / SSM layers of hybrid
    models such as Qwen3.5 and Qwen3-Next) and saturated sliding-window caches.
  - `perSlotContext` / `perSlotContextMax`: the exact state of one slot at the default context (4096 tokens, or the model's maximum
    when smaller) and at the model's maximum context.
  - Need for `s` slots of `c` tokens: `weights + overhead + s * (stateBytesPerSlot + kvBytesPerToken * c)`
    (`MemoryRequirement.TotalFor(slots, contextTokens)`); exact at long contexts, an upper bound at short ones (sliding windows).
    Manifests without `kvBytesPerToken` (built before these fields) scale `perSlotContext` linearly. The figures are a first-pass
    filter: the engine's own footprint estimate on the actual device decides at load time.
- `kernels[]`: the package's weight types (engine names: `q4_k`, `iq2_xxs`, `mxfp4`, `nvfp4`, `f8e4m3_b128`, `bf16`, ...), largest
  share first, with `bytes`, `nativeFeatures` (neutral device feature names, `Sharplight.Runtime.DeviceFeatures`, whose kernels
  accelerate the type; any one helps) and `fallback`. With `fallback: true` (every type the engine has today) every provider (CPU,
  Vulkan, CUDA, Metal) runs the type with its generic kernel when the device lacks the native features: same memory, slower. A host
  shows "native" when the device reports one of `nativeFeatures` for the largest types and "fallback" otherwise; a type with
  `fallback: false` would make a device without its features incompatible. Current mapping: FP8 (`f8e4m3*`, `f8e5m2`):
  `native_gemm_fp8`, `matrix_fp8`, `fp8`, `matrix_f16`; MXFP4: `native_gemm_moe`, `matrix_f16`; NVFP4 and the GPTQ / AWQ integer
  types (`q4z_*`, `q8z_*`): `int_dot`, `matrix_f16`; F16 / BF16: `native_gemm_f16`, `library_gemm`, `matrix_f16`, `matrix_bf16`; GGUF
  block types: `matrix_f16`.
- `deviceFeatures`: features a device must have to run the package at all (empty for every package today).
- `disk`: bytes of the `.slm` (the download is written to `<file>.part` and renamed, so the size is needed once).
- `engine`, `server`, `packageFormat`: versions, checked by `PackageCompatibility`.

## Signing and keys

- ECDSA P-384 with SHA-384 (`ES384`), detached JWS compact signatures (`BASE64URL(header)..BASE64URL(signature)`, RFC 7515
  appendix F). The header names the key id (`sha256:` + 32 hex digits of the SHA-256 of the SubjectPublicKeyInfo).
- Keys (minimal TUF-style):

| Key | Lives | Signs |
|---|---|---|
| Root | offline (hardware token or offline machine) | `keys/keyset.json`: the feed keys with validity and revocations |
| Feed key | the publishing environment's secret store or KMS | `index.json` and every `manifest.json` |

  Hosts ship or configure the root public keys. A new feed key or a revocation is a new key set signed by a root; a manifest is
  checked against the feed keys valid at its publication time (revoked keys are never accepted).
- Key files are PEM (`PRIVATE KEY`, `EC PRIVATE KEY`, `PUBLIC KEY`) or JSON (`{ keyId, algorithm, publicKey, privateKey? }`, SPKI and
  PKCS#8 in base64). Private key files never belong in a
  repository; tests generate their own keys.
- Verification by a host: key set against a trusted root → index signature, expiry, sequence → catalog and page digests → package
  index digest → manifest digest and signature → `.slm` chunk hashes while downloading and the file's SHA-256 at the end (and again
  before the first load).

## Library: Sharplight.ModelFeed

| Type | Purpose |
|---|---|
| `PackageManifest` and parts, `FeedServiceIndex`, `FeedCatalogPage`, `FeedPackageIndex`, `FeedJson` | documents (source-generated JSON) |
| `FeedClient` (`FeedTransports.Create`: `HttpFeedTransport`, `FolderFeedTransport`) | `GetSnapshotAsync` (all checks above, records the sequence in an `IFeedStateStore`), `GetPackageIndexAsync`, `GetManifestAsync` → `VerifiedManifest` |
| `FeedClientOptions` | `Channel`, trusted roots (several for key rotation), `RequireSignature` (default on), `AllowExpiredIndex` (offline), document size limit |
| `DependencyResolver`, `FeedCandidates`, `PackageCompatibility` | highest (or preferred, e.g. installed) version per id satisfying every range, host compatibility, not yanked; required and selected optional dependencies; conflicts reported, nothing partial |
| `SemanticVersion`, `VersionRange` | SemVer 2 precedence; ranges as NuGet intervals (`[1.0.0,2.0.0)`, `1.0.0` = at least) or comparator sets (`>=1.0.0 <2.0.0`) |
| `PackageDownloader` | download with resume (complete chunks of a `.part` file are re-checked and kept), chunk verification, fallback to the next source, final SHA-256 |
| `FeedPublisher`, `FeedVerifier`, `FeedMirror`, `PackageFileHasher` | write and sign a feed folder (`PublishAsync` into a channel, `PromoteAsync` between channels, `SetStateAsync`, `ReindexAsync`), check a channel, copy channels as an offline bundle |

`services.AddModelFeed()` registers the stateless services; hosts create one `FeedClient` per configured feed.

```csharp
var client = new FeedClient(FeedTransports.Create("https://download.sharplight.dev/models/", http),
    new FeedClientOptions { Channel = FeedChannels.Stable, TrustedRoots = [FeedPublicKey.Load("sharplight-feed.pub.pem")] },
    new FileFeedStateStore("feeds.json"));
var snapshot = await client.GetSnapshotAsync();
var plan = await resolver.ResolveAsync([new PackageRequest("whisper-tiny.q5-1-whispercpp")], FeedCandidates.From(client, snapshot),
    new ResolutionOptions { Environment = new CompatibilityEnvironment { EngineVersion = SemanticVersion.Parse("0.1.0") } });
foreach (var package in plan.Packages)
{
    var manifest = await client.GetManifestAsync(snapshot, package.Id, package.Version);
    await downloader.DownloadAsync(manifest.Manifest, [client.Transport], Path.Combine(store, $"{package.Id}.{package.Version}.slm"));
}
```

## Building a feed: feedtool

```
feedtool keygen   --out keys/root                                     # root.key.json (offline) + root.pub.json
feedtool keygen   --out keys/feed
feedtool keyset   --feed out/ --root-key keys/root.key.json --feed-key keys/feed.pub.json [--revoke <kid>] [--version 2]
feedtool plan     [--feed out/]                                       # variants without a package for their pinned revision
feedtool build    --entry qwen3-0.6b [--variant q8-0-gguf] (--out staging/ | --work-dir work/) [--feed out/] [--store <model store>] [--keep-store]
feedtool publish  staging/<id>-<version> ... --feed out/ --key keys/feed.key.json --channel unstable [--validity-days 30] [--move]
feedtool promote  <id>[@<version>] ... --feed out/ --key keys/feed.key.json [--to stable] [--from unstable]   # no file copies
feedtool yank|deprecate|unlist <id> <version> --feed out/ --key keys/feed.key.json --channel <channel> --reason ...
feedtool reindex  --feed out/ --key keys/feed.key.json --channel <channel>   # new sequence and expiry, no content change
feedtool verify   --feed <url|folder> --channel stable [--root keys/root.pub.json] [--files]   # default root: the official key
feedtool mirror   --feed <url|folder> --channel stable [--root ...] --out bundle/ [--filter <id prefix>]   # once per channel
```

(`dotnet run --project tools/model-feed/Sharplight.FeedTool -- <command>`.)

- `build` downloads exactly the pinned files of the variant and its companions (checksums verified), converts them with companions
  merged, opens the result, hashes and chunks it and writes `manifest.json` from the catalog entry (license, publisher, verification,
  provenance) and the overlay `tools/model-feed/feed-overlay.json` (feed-only data: dependencies, summaries, languages, minimum host
  version, listing state; `models` holds defaults for every variant of a catalog entry, `packages` per-package entries that override
  them field by field).
- Versions are chosen by the tool against the stored versions of the feed (all channels): first build 1.0.0; another pinned revision `MAJOR`+1; the same pins rebuilt
  `MINOR`+1; the same package file with a different manifest (compared without version, file name, build time, converter and
  minimum engine, which a rebuild changes by itself) is a manifest-only `PATCH`+1 that reuses the file: the staging folder holds
  `manifest.json` and `reuse.json` (`{"version","file"}` of the earlier version), `publish` checks size and hashes against the earlier
  manifest and, on object storage, the workflow copies the earlier object server-side; an identical file with identical metadata is
  not built again. `promote <id>` then promotes the new latest version.
- `build --work-dir` keeps the store (`<work-dir>/store`) and the staging root (`<work-dir>/staging`) on one volume. Unless
  `--keep-store` is given (or the model was in the store before), the converted package is moved, not copied, into the staging folder
  and the model's store folder is removed after each entry, so a build needs disk for one entry's sources plus its package. When the
  feed already stores the exact package, the staging folder holds only the stored manifest: publishing it lists the version in a
  channel that misses it (a run interrupted after the package objects were stored) and reports it unchanged otherwise.
- `publish` signs each manifest and the new index; published versions are immutable (publishing other content under an existing
  version fails). `--move` moves the package files into the feed folder instead of copying them. `keyset` needs the root key,
  nothing else does.
- The output folder is uploaded as is (package objects before `index.json`).

## Publishing the official feed

The official feed lives in the Cloudflare R2 bucket of the Studio downloads under `models/`: https://download.sharplight.dev/models/
(channels `stable/index.json` and `unstable/index.json` there; hosts default to stable). It is published by `.github/workflows/model-feed.yml` (a push to develop that changes `tools/model-feed/publish-request.json`, manual dispatch, plus a weekly re-sign; environment `studio-release`, the
same R2 secrets as `studio.yml`):

**Publish requests.** Until the workflow is on the default branch, the trigger is a commit to develop that changes
`tools/model-feed/publish-request.json`:

```json
{
  "entries": ["qwen3-0.6b:q8-0-gguf", "silero-vad-v5:onnx"],
  "channel": "stable",
  "promote": ["qwen3-0.6b.q8-0-gguf@1.0.0", "silero-vad-v5.onnx"],
  "validityDays": 30,
  "reason": "why this publication"
}
```

`entries` are `entry` or `entry:variant` ids of the verified catalog (default variant: the recommended one); they are built and
published to `unstable`. `promote` lists package versions (`id@version`, or `id` for its latest version in unstable) to add to
`channel`, which is `stable` (promotions go from unstable to stable). With neither, the run only re-signs. Manual dispatch (inputs
`entries`, `promote`, `validity_days`) and the schedule work once the workflow is on the default branch. A run:

A run has three jobs, one publication at a time (concurrency group `model-feed`, never cancelled halfway):

1. **plan** (ubuntu-latest, no secrets): reads the request, and `tools/model-feed/split-request.py` splits the entries by the size of
   their pinned files in the catalog (variant plus companions): above the repository variable `MODEL_FEED_LARGE_BYTES` (default
   6 GB: the hosted runner has about 14 GB free and a build needs the sources plus the package, about twice that) or listed in
   `MODEL_FEED_LARGE_ENTRIES` (entry, entry:variant or package ids), an entry is large. Promotions of packages that the large pass
   builds go with it.
2. **small** (ubuntu-latest) and then **large** (the self-hosted runner, see "Self-hosted feed runner"), both
   `.github/workflows/model-feed-publish.yml` in the `studio-release` environment (the only place the signing key and R2 secrets
   come from). Each pass:
   1. fetches the published listings (no package files), writes a key set if the bucket has none;
   2. per entry: builds it (`entry` or `entry:variant`; versions are chosen against the published feed, an identical package is not
      built again, a changed catalog kind, license or description republishes as the next patch version), publishes it into
      `unstable` in the local feed folder (the file is moved), uploads its objects to `models/packages/<id>/<version>/` (the package
      file first, then `manifest.json` and `manifest.sig`; `Cache-Control: public, max-age=31536000, immutable`;
      `tools/model-feed/upload-package.sh`) and deletes the local package file. An object that exists with the same content (package
      files carry their SHA-256 as object metadata `sha256`) is kept, so a rerun after an interruption resumes; one with other content
      fails the run (published objects are never replaced). A multipart upload becomes an object only when complete;
   3. promotes the listed versions into `stable`, re-signs both channels' indexes (so every run refreshes both expiries) and verifies
      both channels against the official public key;
   4. uploads per channel (`models/unstable/`, then `models/stable/`) the package indexes, catalog pages, `catalog/index.json`,
      `index.json` and `index.sig` (`no-cache`). Nothing is ever deleted.

   The large pass runs only after the small one succeeded, for a push or a dispatch on develop, when the repository variable
   `MODEL_FEED_SELF_HOSTED` is `true`; when the optional secret `MODEL_FEED_RUNNER_TOKEN` (a token that may list runners: fine-grained
   "Administration: read" on the repository, or "Self-hosted runners: read" on the organization for an organization runner) is set,
   the plan also checks that a runner labelled `sharplight-feed` is online. Otherwise the large entries (and their promotions) are
   skipped with a warning and the small ones publish as usual. Without the token, an enabled but offline runner leaves the large pass
   queued (GitHub cancels it after 24 hours); the small pass is published before it either way.

The first publication (`tools/model-feed/publish-request.json` as committed) builds the five first-feed packages into unstable and
promotes all of them, so both channels list the same five packages.

Without the secret `MODEL_FEED_SIGNING_KEY` (or with incomplete R2 secrets) the workflow builds nothing and says so in a warning.

An index expires after `validity_days` (default 30) and hosts reject an expired index (freeze protection). A **weekly scheduled run**
(Mondays 04:17 UTC) therefore only re-signs the index: no builds, a new sequence and a new expiry 30 days ahead. Any run without
entries does the same, and does nothing before the first publication.

**Until the next release** the weekly re-sign does not run: schedules (like dispatch) only run for workflows on the default branch
(main), and the `studio-release` environment admits only develop and release/*. Until then, re-sign at least every three weeks by
committing to develop a `publish-request.json` with empty `entries` and `promote` lists (and a new `reason`, so the file changes).

### Publishing keys

The official feed uses a single-key setup: one ECDSA P-384 key signs the key set (as root) and the index and manifests (as the
only feed key). Its public key is committed as `tools/model-feed/keys/sharplight-feed.pub.pem`; hosts trust it by default and
`feedtool verify` / `mirror` use it when no `--root` is given.

Setting up a key (done once by the product owner; never by CI or an agent, and the private key is never committed):

1. Generate the key pair on a trusted machine:
   `openssl ecparam -name secp384r1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out feed.key.pem` and
   `openssl pkey -in feed.key.pem -pubout -out feed.pub.pem` (or `feedtool keygen --out feed`, JSON key files).
2. Store the private key as the secret `MODEL_FEED_SIGNING_KEY` of the GitHub environment `studio-release` (the whole PEM text),
   then delete or lock away the local private key file.
3. Commit the public key as `tools/model-feed/keys/sharplight-feed.pub.pem` and add it to the hosts' trusted keys (for a server:
   the trusted root keys of its feed source configuration; Studio: its built-in feed trust).

Rotation without downtime:

1. Generate the new key pair; add the new public key to the hosts' trusted keys **beside** the old one (hosts accept a key set
   signed by any trusted root) and ship that configuration.
2. When the hosts have it, replace the secret with the new private key, delete `models/keys/keyset.json` and `keyset.sig` from the
   bucket (the next run writes a key set signed by the new key) and run the workflow with no entries: the index is re-signed. Old
   manifests stay valid as long as the key set lists their key: for a full re-sign of existing manifests, publish new versions.
3. Remove the old public key from the hosts' trusted keys in a later release.

A separate offline root (TUF-style) is supported by the format: write the key set with `feedtool keyset --root-key <offline root>
--feed-key <feed public key>` and upload it to `models/keys/`; the workflow then only signs with the feed key, and hosts trust the
root (set the repository variable `MODEL_FEED_ROOT_PUBLIC_KEY` to the root's public key PEM so the workflow verifies against it).

## Self-hosted feed runner

Packages above about 6 GB (the large catalog variants are 18 to 111 GB) do not fit the disk of a GitHub-hosted runner, so the
workflow's large pass runs on a self-hosted runner with the labels `self-hosted` and `sharplight-feed` (registered with
`self-hosted, linux, x64, sharplight-feed`).

**What a build needs** (measured with `feedtool build --work-dir`, 2026-11-20; times include the Hub download on a fast link):

| Package | Sources | Package | Peak disk | Peak RAM (anonymous) | Time |
|---|---|---|---|---|---|
| Qwen3 0.6B Q8_0 GGUF | 0.64 GB | 0.65 GB | 0.6 GiB | 89 MiB | 8 s |
| Qwen3.5 9B Q4_K_M GGUF | 5.7 GB | 6.6 GB | 12.3 GiB | 91 MiB | 29 s |
| Qwen3.5 4B BF16 safetensors | 9.3 GB | 9.4 GB | 17.4 GiB | 159 MiB | 91 s |
| gpt-oss-20b MXFP4 GGUF | 12.1 GB | 12.1 GB | 22.6 GiB | 116 MiB | 57 s |
| gpt-oss-20b MXFP4 safetensors (repacked weights) | 13.8 GB | 13.8 GB | 25.0 GiB | 2.4 GiB | 119 s |

- Disk: the sources plus the package, about twice the package (the sources are deleted after conversion; the package is moved, not
  copied, into staging and into the feed folder, and deleted after its upload). The feed listings without package files are a few
  MB. Extrapolated: 18 GB → about 37 GB, 63 GB (gpt-oss-120b) → about 127 GB, 111 GB → about 225 GB. Give the work directory at least
  twice the largest package; about 250 GB of working space covers the current catalog.
- RAM: GGUF and plain safetensors are memory-mapped and streamed into the package (the file-backed resident memory is page cache, which
  the kernel reclaims). Weights the importer derives (repacked, dequantized or fused, e.g. MXFP4 safetensors) are produced one at a
  time and released after they are written, so the peak is the largest derived weight plus a little: 16 GB of RAM is enough.
- Time: the download (the Hub link), conversion at a few seconds per GB, two SHA-256 passes over the package (build and publish) and
  the upload to R2 (the uplink; multipart with 128 MB parts, 8 in parallel). A 111 GB package takes tens of minutes on a fast link and
  some hours on a slow uplink; the large pass has a 24-hour job timeout.
- No GPU: the build converts and opens the package with the package reader only (no weights are loaded, `verification.packageCheck`
  is `{ "loaded": true }` from that check), and the job hides GPUs (`CUDA_VISIBLE_DEVICES` empty), so a GPU shared with other work is
  not touched. A load check on a device is not part of the feed build.

**Installing a runner** (once per machine, by the owner; `tools/model-feed/runner/setup-runner.sh` does all of it):

1. Get a registration token: the repository's (or, for an organization runner, the organization's) Settings → Actions → Runners →
   New self-hosted runner. It expires after an hour. Agents and CI never fetch or create tokens.
2. On the machine, as root: `RUNNER_URL=https://github.com/<organization or owner/repository> RUNNER_TOKEN=<token>
   tools/model-feed/runner/setup-runner.sh --user ghrunner --feed-work /home/ghrunner/feed-work`. It creates the non-root user (no
   sudo), installs jq, git, curl, python3, unzip and the AWS CLI v2, registers the runner with the labels above and installs it as a
   systemd service under that user. .NET is installed by `actions/setup-dotnet` in each job into the runner's tool cache (no root).
3. On GitHub: for an organization runner, allow this public repository in the runner group (Organization settings → Actions →
   Runner groups → the group → Repository access; public repositories must be allowed explicitly) and, where the plan offers it,
   restrict the group to the workflow `model-feed.yml` on `refs/heads/develop`. Then set the repository variables
   `MODEL_FEED_SELF_HOSTED=true` and `MODEL_FEED_WORK_DIR=/home/ghrunner/feed-work` (default: next to the job's workspace), and
   optionally the secret `MODEL_FEED_RUNNER_TOKEN` for the online check.

**Security.** The repository is public: a self-hosted runner must only run trusted workflows of trusted branches, never code from pull
requests.

- `model-feed.yml` has no `pull_request` trigger; the large job runs only for `push` and `workflow_dispatch` on develop, in the
  `studio-release` environment (which admits only develop and release/*, and holds the signing key and R2 credentials).
- `setup-runner.sh` installs a job-started hook (`ACTIONS_RUNNER_HOOK_JOB_STARTED`) that fails every job on the machine whose event is
  not `push` or `workflow_dispatch`, whose ref is not `refs/heads/develop`, or whose workflow is not `model-feed.yml` of the
  repository, so a workflow added in a fork's pull request cannot run there even if it names the labels. A runner registered by hand
  should get the same hook (copy `feed-job-guard.sh` and the `.env` line from the script).
- Keep "Require approval for all outside collaborators" for fork pull request workflows in the repository's Actions settings, and do
  not give the runner user sudo, cloud credentials or other secrets: the job receives its secrets from the environment only.
- The runner is not ephemeral: the work directory keeps the model store (so an interrupted download resumes) but nothing secret;
  the signing key is written to the job's temporary directory and removed at the end of every job.
