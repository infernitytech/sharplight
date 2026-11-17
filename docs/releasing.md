# Releasing

## Branches

- `main` is stable. `develop` is the unstable integration branch every working branch merges into; it merges into `main` once
  everything is green.
- Releases are cut from `release/vX.Y` branches with [Nerdbank.GitVersioning](https://github.com/dotnet/Nerdbank.GitVersioning)
  (`version.json` has `release.branchName = release/v{version}`).
- Versions: builds of `main` and `release/v*` are public releases (`0.1.7`, or `0.1.7-alpha` while `version.json` still says
  `0.1-alpha`); builds of `develop` and everything else are prereleases with the commit id (`0.2.3-alpha.g1a2b3c4`).

## What CI does

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | push to `develop`, `main`, `release/**`; pull requests | restore, build, run every test suite that works on a hosted runner (not the Vulkan and ONNX conformance suites) |
| `packages.yml` | same | pack the NuGet libraries (artifact). **Publishes to nuget.org only on `release/**` pushes** (trusted publishing, environment `nuget`, variable `NUGET_USER`); never from `develop`, `main` or a pull request. `develop` pushes also publish the prerelease packages to the GitHub Packages feed once CI is green for that commit (see [Prerelease feed](#prerelease-feed)) |
| `studio.yml` | same (not for documentation-only changes) | Studio packages for 5 runtimes on native runners. `develop` push: upload to R2, channels `<rid>-unstable`. `release/**` push: upload to R2, channels `<rid>-stable`, and attach the installers to the GitHub release `v<version>`. `main` and pull requests: build only. Job `catalog`: validates `catalog/verified-models.json` and, on `develop` pushes, uploads it to R2 as `catalog/verified-models.json` (docs/verified-catalog.md) |

## Release procedure

1. **Prepare.** On a clean, up-to-date `develop` (green CI):
   ```sh
   nbgv prepare-release          # creates release/v0.1 and bumps develop's version.json to the next minor (0.2-alpha)
   git push origin release/v0.1 develop
   ```
   (Install it with `dotnet tool install -g nbgv`.) With `firstUnstableTag: alpha` the release branch's version.json gets `0.1`
   (the `-alpha` suffix is removed); `develop` moves on.
2. **Stabilize.** Only fixes go onto `release/v0.1` (cherry-pick or merge from `develop`). Every push builds and tests,
   publishes the NuGet packages `0.1.N` (N = git height; `--skip-duplicate`) and the Studio packages on the `*-stable` channels.
   Installed stable Studio builds offer the update.
3. **Publish.** The last green push of the release branch is the release: the NuGet packages are on nuget.org, the GitHub release
   `v0.1.N` has the Studio installers (the tag is created by the workflow, never by hand), and R2 has the update feeds. Nothing
   else to do. Check the Actions run, then the GitHub release and `releases.<rid>-stable.json` at the update URL.
4. **Merge back.** Merge `release/v0.1` into `main` (this makes `main` the stable line, and a public release version), and merge
   it into `develop` so fixes made on the branch are not lost (`git merge release/v0.1` on `develop`; resolve the `version.json`
   conflict in favor of `develop`'s).

Hotfixes: fix on `release/vX.Y`, push, merge back as in step 4.

## Prerelease feed

Every push to `develop` whose CI run (`ci.yml`) succeeds publishes all packable projects (the libraries under `src/`) to the
GitHub Packages NuGet feed of the repository owner, with the Nerdbank.GitVersioning prerelease version of that commit
(`0.2.3-alpha.g1a2b3c4`; the same version the package artifact has). Job `prerelease` in `packages.yml` waits for the CI run of the
same commit, then pushes with the workflow's `GITHUB_TOKEN` (`packages: write`, `--skip-duplicate`). Forks, pull requests and other
branches never publish; a CI run cancelled by a newer push publishes nothing (the newer push publishes its own build). nuget.org
behaviour is unchanged: only `release/**` pushes go there.

Feed URL: `https://nuget.pkg.github.com/infernitytech/index.json`. GitHub Packages requires authentication for every restore, also
for packages of a public repository.

**Consuming it from another project.** Add the feed next to nuget.org and map the `Sharplight*` packages to it, so nothing else is
ever resolved from it:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
    <add key="sharplight-prerelease" value="https://nuget.pkg.github.com/infernitytech/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="sharplight-prerelease">
      <package pattern="Sharplight" />
      <package pattern="Sharplight.*" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
```

Pin exact versions (`0.2.3-alpha.g1a2b3c4`) rather than floating ranges; the list of published versions is on the package pages of
the organization (Packages tab) or `dotnet package search Sharplight --source sharplight-prerelease --prerelease`.

- **GitHub Actions** (a repository of the same organization): give the consuming repository read access once, in each package's
  settings under *Manage Actions access* (or set the packages' default access in the organization settings), add
  `permissions: packages: read` to the workflow and pass the workflow token before restoring:
  ```sh
  dotnet nuget update source sharplight-prerelease --username "$GITHUB_ACTOR" --password "$GITHUB_TOKEN" --store-password-in-clear-text
  ```
- **Local development**: create a personal access token (classic) with only the `read:packages` scope (fine-grained tokens are not
  supported by the NuGet registry), then store it in the user-level NuGet configuration, never in the repository:
  ```sh
  dotnet nuget update source sharplight-prerelease --username <github-user> --password <token> --configfile ~/.nuget/NuGet/NuGet.Config
  ```
  (use `dotnet nuget add source <feed url> --name sharplight-prerelease ...` if the source is not defined in that file yet).
  Alternatively the consuming repository's `nuget.config` can read the credentials from environment variables, which works the
  same locally and in CI:
  ```xml
  <packageSourceCredentials>
    <sharplight-prerelease>
      <add key="Username" value="%SHARPLIGHT_FEED_USER%" />
      <add key="ClearTextPassword" value="%SHARPLIGHT_FEED_TOKEN%" />
    </sharplight-prerelease>
  </packageSourceCredentials>
  ```

## Studio update channels and Cloudflare R2

Ten channels: `<rid>-stable` and `<rid>-unstable` for `win-x64`, `win-arm64`, `linux-x64`, `linux-arm64`, `osx-arm64`. The package
job fetches the channel's previous release (`vpk download s3`) so Velopack can build delta packages, packs, then uploads
(`vpk upload s3`; unstable keeps the last 10 releases). All channels share one folder (`<prefix>/` in the bucket), each with its
own `releases.<channel>.json` feed. The apps read `STUDIO_UPDATE_URL`, which points at that folder.

One-time setup on GitHub:

| Where | Name | Value |
|---|---|---|
| repository variable | `STUDIO_UPDATE_URL` | public URL of the folder, `https://download.sharplight.dev/studio` |
| repository variable | `STUDIO_UPDATE_PREFIX` | key prefix in the bucket, `studio` (default when unset) |
| repository variable | `NUGET_USER` | nuget.org user for trusted publishing (already used) |
| environment `studio-release` (restrict it to `develop` and `release/*`) | secret `R2_ACCOUNT_ID` | Cloudflare account id (endpoint `https://<id>.r2.cloudflarestorage.com`, region `auto`) |
| same environment | secrets `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | R2 API token with Object Read and Write on the bucket |
| same environment | secret `R2_BUCKET` | bucket name |
| environment `nuget` | (already exists) | trusted publisher for `packages.yml` |

Only the uploading pushes (develop, release/*) enter the `studio-release` environment; pull requests and other branches build
without any secret.

**R2 and cache settings.** Serve the bucket on the custom domain (public access through the domain, not the `r2.dev` URL). Cache
rules on that domain:

- `releases.*.json`, `RELEASES-*` and `assets.*.json` (the feeds, rewritten on every upload): no cache or a very short one
  (for example Edge TTL 1 minute, Browser TTL "respect origin" with `Cache-Control: no-cache`), otherwise clients see updates late.
- `*.nupkg`, installers and archives (file names contain the version and are never rewritten): immutable, long edge and browser
  TTL (`Cache-Control: public, max-age=31536000, immutable`).

## Code signing (not set up yet)

Unsigned packages work but trigger SmartScreen (Windows) and Gatekeeper (macOS) warnings. The hooks exist and stay unused until the
secrets exist:

- Windows Authenticode: secret `WINDOWS_SIGN_TEMPLATE`, a `vpk --signTemplate` command (Azure Trusted Signing or signtool).
- macOS Developer ID and notarization: secrets `MACOS_SIGN_APP_IDENTITY`, `MACOS_SIGN_INSTALL_IDENTITY`, `MACOS_NOTARY_PROFILE`; add a
  workflow step that imports the certificate into a temporary keychain and stores the notarytool profile first. Confirm the flag
  names with `vpk [osx] pack -H` on the runner; they are passed by `packaging/pack-studio.sh`.

## Building a package locally

```sh
dotnet tool install vpk --tool-path ~/tools --version 1.2.161       # needs the .NET 10 runtime, or DOTNET_ROLL_FORWARD=Major
VPK=~/tools/vpk packaging/pack-studio.sh linux-x64 linux-x64-unstable 0.1.99-alpha.gdeadbee out
```

The Linux AppImage needs `squashfs-tools` (`mksquashfs`) on the machine that packs. Set `STUDIO_UPDATE_URL` to bake an update feed
into the build.
