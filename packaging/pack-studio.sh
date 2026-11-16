#!/usr/bin/env bash
# Publishes Sharplight Studio self-contained for one runtime and packs it with Velopack (vpk) for one update channel.
#
# Usage: packaging/pack-studio.sh <rid> <channel> <version> <output dir> [release notes file]
#   rid      win-x64, win-arm64, linux-x64, linux-arm64 or osx-arm64 (run it on a runner of that OS and architecture)
#   channel  <rid>-stable or <rid>-unstable; installed apps only look for updates in the channel they were packed for
#   version  SemVer 2 (NuGetPackageVersion of Nerdbank.GitVersioning)
#   output   Velopack release directory. Fill it with `vpk download s3` first to get delta packages against the previous release.
#
# Environment (all optional):
#   VPK                      the vpk command (default: vpk)
#   STUDIO_UPDATE_URL        update feed URL baked into the app (see docs/studio.md, "Installing and updating")
#   STUDIO_WORK              scratch directory for the publish output (default: a temporary directory, removed at the end)
#   VPK_SIGN_TEMPLATE        Windows code signing command template for vpk (Authenticode), unused until a certificate exists
#   MACOS_SIGN_APP_IDENTITY, MACOS_SIGN_INSTALL_IDENTITY, MACOS_NOTARY_PROFILE
#                            Developer ID identities and notarytool keychain profile, unused until the certificates exist
set -euo pipefail

rid="${1:?runtime identifier required}"
channel="${2:?channel required}"
version="${3:?version required}"
out="$(mkdir -p "${4:?output directory required}" && cd "$4" && pwd)"
notes="${5:-}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
vpk="${VPK:-vpk}"

work="${STUDIO_WORK:-}"
if [ -z "$work" ]; then
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
fi
mkdir -p "$work"
publish="$work/publish"
rm -rf "$publish"

props=()
if [ -n "${STUDIO_UPDATE_URL:-}" ]; then
    props+=("-p:StudioUpdateUrl=$STUDIO_UPDATE_URL")
fi

# Trimming and native AOT stay off: Avalonia compiled bindings, System.CommandLine and the engine's reflection are not trim-safe yet.
dotnet publish "$repo/apps/Sharplight.Studio/Sharplight.Studio.csproj" -c Release -r "$rid" --self-contained \
    -p:PublishReadyToRun=true ${props[@]+"${props[@]}"} -o "$publish"

# Sharplight.Audio's native library (built by native.yml into native/sharplight_audio/out/<rid>/) rides along under runtimes/<rid>/native/.
if ! ls "$publish/runtimes/$rid/native/"*sharplight_audio* >/dev/null 2>&1; then
    echo "the native audio library for $rid is missing from the publish output (build it with native/sharplight_audio/build.sh or download the native-audio artifacts into native/sharplight_audio/out)" >&2
    exit 1
fi

# Sharplight.Video's native library (libvpx, built by native.yml into native/sharplight_video/out/<rid>/) rides along the same way.
if ! ls "$publish/runtimes/$rid/native/"*sharplight_video* >/dev/null 2>&1; then
    echo "the native video library for $rid is missing from the publish output (build it with native/sharplight_video/build.sh or download the native-video artifacts into native/sharplight_video/out)" >&2
    exit 1
fi

cp "$repo/src/Sharplight.Audio/THIRD-PARTY-NOTICES.md" "$publish/THIRD-PARTY-NOTICES-Audio.md"
cp "$repo/src/Sharplight.Video/THIRD-PARTY-NOTICES.md" "$publish/THIRD-PARTY-NOTICES-Video.md"

args=(--packId SharplightStudio --packVersion "$version" --packDir "$publish" --runtime "$rid" --channel "$channel"
    --outputDir "$out" --packTitle "Sharplight Studio" --packAuthors "Sharplight contributors")
if [ -n "$notes" ]; then
    args+=(--releaseNotes "$notes")
fi

case "$rid" in
    win-*)
        # Setup.exe and a portable zip. Signing: vpk reads VPK_SIGN_TEMPLATE itself (for example an Azure Trusted Signing or
        # signtool command line); nothing is signed while it is unset.
        "$vpk" pack "${args[@]}" --mainExe sharplight-studio.exe --icon "$repo/apps/Sharplight.Studio/Assets/Icons/sharplight-studio.ico" \
            --shortcuts Desktop,StartMenuRoot
        ;;
    linux-*)
        # AppImage (the Velopack package that updates itself) plus a plain tar.gz of the same self-contained folder.
        "$vpk" pack "${args[@]}" --mainExe sharplight-studio --icon "$repo/packaging/linux/sharplight-studio.png" \
            --categories "Development;Science;ArtificialIntelligence"
        tar -C "$publish" -czf "$out/sharplight-studio-$version-$rid.tar.gz" .
        ;;
    osx-*)
        # .app bundle in a Velopack .pkg installer plus a zip. The Vulkan provider runs through MoltenVK: a publish on macOS
        # puts libMoltenVK.dylib (Sharplight.Vulkan.MoltenVK, docs/macos.md) next to the executable, where the provider finds it.
        if [ ! -f "$publish/libMoltenVK.dylib" ]; then
            echo "libMoltenVK.dylib is missing from the publish output (publish on macOS, or pass -p:SharplightBundleMoltenVK=true)" >&2
            exit 1
        fi
        cp "$repo/src/Sharplight.Vulkan.MoltenVK/THIRD-PARTY-NOTICES.md" "$publish/THIRD-PARTY-NOTICES-MoltenVK.md"
        cp "$repo"/src/Sharplight.Vulkan.MoltenVK/obj/moltenvk/*/LICENSE-MoltenVK.txt "$publish/LICENSE-MoltenVK.txt"
        iconset="$work/sharplight-studio.iconset"
        mkdir -p "$iconset"
        for size in 16 32 128 256; do
            cp "$repo/apps/Sharplight.Studio/Assets/Icons/sharplight-studio-$size.png" "$iconset/icon_${size}x${size}.png"
        done
        iconutil -c icns "$iconset" -o "$work/sharplight-studio.icns"
        mac=(--mainExe sharplight-studio --icon "$work/sharplight-studio.icns" --bundleId com.infernitytech.sharplight.studio)
        # Developer ID signing and notarization hooks: unused (the bundle stays unsigned or ad-hoc signed) until the user provides
        # the certificates as secrets. Check the flag names with `vpk [osx] pack -H` on the macOS runner before enabling them.
        if [ -n "${MACOS_SIGN_APP_IDENTITY:-}" ]; then
            mac+=(--signAppIdentity "$MACOS_SIGN_APP_IDENTITY")
        fi
        if [ -n "${MACOS_SIGN_INSTALL_IDENTITY:-}" ]; then
            mac+=(--signInstallIdentity "$MACOS_SIGN_INSTALL_IDENTITY")
        fi
        if [ -n "${MACOS_NOTARY_PROFILE:-}" ]; then
            mac+=(--notaryProfile "$MACOS_NOTARY_PROFILE")
        fi
        "$vpk" pack "${args[@]}" "${mac[@]}"
        ;;
    *)
        echo "Unsupported runtime identifier: $rid" >&2
        exit 2
        ;;
esac

ls -la "$out"
