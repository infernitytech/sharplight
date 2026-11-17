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
  engine, graphs, added portable code, higher minimum engine); `PATCH`: manifest-only change. Prerelease versions are only chosen when
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
    "deviceFeatures": [],         // neutral engine feature names a device must have
    "memory": [ { "device": "gpu", "weights": 652000000, "overhead": 480000000,
                  "perSlotContext": { "tokens": 8192, "bytes": 310000000 } } ],
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
| `requirements.memory` | first-pass placement filter; the engine's own estimate on the device decides |
| `dependencies` | NuGet-style ranges. `required`: installed together, a missing one fails the install. `optional`: offered. Roles are auxiliary roles (`draft`, `voice-activity`, `lora`, ...) or `pairing`. Companions merged into the `.slm` are not dependencies. |
| `license.gated` | the host asks for explicit acceptance of the terms before install |
| `files[].chunks` | SHA-256 per chunk: resumable, verified transfers from several sources (mirrors, peers) |
| `state` | the state at publication; the current state (deprecated, yanked, unlisted) is in the package index |

Readers ignore unknown fields; `schemaVersion` changes only for breaking changes.

## Feed layout

```
index.json, index.sig                          service index and its signature
keys/keyset.json, keys/keyset.sig              feed keys, signed by a root key
catalog/index.json                             list of pages
catalog/page/<n>.json                          package summaries, 200 per page, ordered by id
packages/<id>/index.json                       all versions of a package: manifest digest, size, publication time, current state,
                                               requirements and dependencies (what resolution needs)
packages/<id>/<version>/manifest.json, manifest.sig, <id>.<version>.slm
```

`index.json`:

```jsonc
{
  "schemaVersion": 1,
  "name": "Sharplight verified models",
  "updated": "2026-11-17T10:05:00Z",
  "expires": "2026-12-17T10:05:00Z",     // freeze protection: an expired index is rejected (accepted with a warning when allowed offline)
  "sequence": 412,                        // strictly increasing; lower than last seen is rejected (rollback protection)
  "resources": { "catalog": "catalog/index.json", "packages": "packages/{id}/index.json", "keys": "keys/keyset.json" },
  "catalog": { "pages": 3, "packages": 412, "digest": "sha256:..." }
}
```

Digests (`sha256:<hex>` over the exact bytes) chain index → catalog index → pages → package index → manifest → `.slm` (and its chunks),
so one signature check on the index authenticates every listing, and the manifest signature authenticates each package on its own:
mirrors and peers need no trust. A page entry carries what a listing needs without further requests: id, latest version, all
versions, display, publisher and country of origin, license, model (kind, tasks, capabilities, size data), requirements,
dependencies, size, state, manifest and package index digests.

Caching: `index.json`, `index.sig`, `catalog/**` and `packages/<id>/index.json` short-lived (`no-cache`, ETag); everything under
`packages/<id>/<version>/` immutable (`max-age=31536000, immutable`). Package files are read with range requests. Publication is
atomic from a reader's view: package objects first, package indexes and pages next, `index.json` and `index.sig` last.

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
- Key files are JSON (`{ keyId, algorithm, publicKey, privateKey? }`, SPKI and PKCS#8 in base64). Private key files never belong in a
  repository; tests generate their own keys.
- Verification by a host: key set against a trusted root → index signature, expiry, sequence → catalog and page digests → package
  index digest → manifest digest and signature → `.slm` chunk hashes while downloading and the file's SHA-256 at the end (and again
  before the first load).

## Library: Sharplight.ModelFeed

| Type | Purpose |
|---|---|
| `PackageManifest` and parts, `FeedServiceIndex`, `FeedCatalogPage`, `FeedPackageIndex`, `FeedJson` | documents (source-generated JSON) |
| `FeedClient` (`FeedTransports.Create`: `HttpFeedTransport`, `FolderFeedTransport`) | `GetSnapshotAsync` (all checks above, records the sequence in an `IFeedStateStore`), `GetPackageIndexAsync`, `GetManifestAsync` → `VerifiedManifest` |
| `FeedClientOptions` | trusted roots, `RequireSignature` (default on), `AllowExpiredIndex` (offline), document size limit |
| `DependencyResolver`, `FeedCandidates`, `PackageCompatibility` | highest (or preferred, e.g. installed) version per id satisfying every range, host compatibility, not yanked; required and selected optional dependencies; conflicts reported, nothing partial |
| `SemanticVersion`, `VersionRange` | SemVer 2 precedence; ranges as NuGet intervals (`[1.0.0,2.0.0)`, `1.0.0` = at least) or comparator sets (`>=1.0.0 <2.0.0`) |
| `PackageDownloader` | download with resume (complete chunks of a `.part` file are re-checked and kept), chunk verification, fallback to the next source, final SHA-256 |
| `FeedPublisher`, `FeedVerifier`, `FeedMirror`, `PackageFileHasher` | write and sign a feed folder, check a whole feed, copy it as an offline bundle |

`services.AddModelFeed()` registers the stateless services; hosts create one `FeedClient` per configured feed.

```csharp
var client = new FeedClient(FeedTransports.Create("https://example.org/models/v1", http),
    new FeedClientOptions { TrustedRoots = [FeedPublicKey.Load("root.pub.json")] }, new FileFeedStateStore("feeds.json"));
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
feedtool build    --entry qwen3-0.6b [--variant q8-0-gguf] --out staging/ [--feed out/] [--store <model store>]
feedtool publish  staging/<id>-<version> ... --feed out/ --key keys/feed.key.json [--validity-days 30]
feedtool yank|deprecate|unlist <id> <version> --feed out/ --key keys/feed.key.json --reason ...
feedtool reindex  --feed out/ --key keys/feed.key.json                # new sequence and expiry, no content change
feedtool verify   --feed <url|folder> --root keys/root.pub.json [--files]
feedtool mirror   --feed <url|folder> --root keys/root.pub.json --out bundle/ [--filter <id prefix>]
```

(`dotnet run --project tools/model-feed/Sharplight.FeedTool -- <command>`.)

- `build` downloads exactly the pinned files of the variant and its companions (checksums verified), converts them with companions
  merged, opens the result, hashes and chunks it and writes `manifest.json` from the catalog entry (license, publisher, verification,
  provenance) and the overlay `tools/model-feed/feed-overlay.json` (feed-only data: dependencies, summaries, languages, minimum host
  version, listing state).
- Versions are chosen by the tool against the feed: first build 1.0.0; another pinned revision `MAJOR`+1; the same pins rebuilt
  `MINOR`+1; an identical package file is not built again.
- `publish` signs each manifest and the new index; published versions are immutable (publishing other content under an existing
  version fails). `keyset` needs the root key, nothing else does.
- The output folder is uploaded as is (package objects before `index.json`).
