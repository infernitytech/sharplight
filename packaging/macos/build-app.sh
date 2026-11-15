#!/usr/bin/env bash
# Builds "Sharplight Studio.app" for Apple Silicon: a self-contained osx-arm64 publish in Contents/MacOS, the bundled
# MoltenVK in Contents/Frameworks, Info.plist and an .icns icon, ad-hoc signed and zipped.
# Usage (on macOS): packaging/macos/build-app.sh <version> [output directory]
#
# The bundle is not signed with a Developer ID nor notarized yet: Gatekeeper asks users to approve it once (docs/macos.md).
set -euo pipefail

version="${1:?version required}"
mkdir -p "${2:-artifacts/studio}"
out="$(cd "${2:-artifacts/studio}" && pwd)"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build-app.sh needs macOS (sips, iconutil, codesign)." >&2
    exit 1
fi

app="$work/Sharplight Studio.app"
contents="$app/Contents"
mkdir -p "$contents/MacOS" "$contents/Resources" "$contents/Frameworks"

# Self-contained publish; on macOS the Vulkan provider brings libMoltenVK.dylib along (Sharplight.Vulkan.MoltenVK).
dotnet publish "$repo/apps/Sharplight.Studio/Sharplight.Studio.csproj" -c Release -r osx-arm64 --self-contained \
    -p:PublishReadyToRun=true -p:UseAppHost=true -o "$contents/MacOS"

# MoltenVK goes to Contents/Frameworks, where the Vulkan provider looks for it (../Frameworks next to the executable).
mv "$contents/MacOS/libMoltenVK.dylib" "$contents/Frameworks/libMoltenVK.dylib"
cp "$repo/LICENSE" "$contents/Resources/LICENSE.txt"
cp "$repo/src/Sharplight.Vulkan.MoltenVK/THIRD-PARTY-NOTICES.md" "$contents/Resources/THIRD-PARTY-NOTICES.md"
cp "$repo"/src/Sharplight.Vulkan.MoltenVK/obj/moltenvk/*/LICENSE-MoltenVK.txt "$contents/Resources/LICENSE-MoltenVK.txt"

# Bundle version: CFBundleShortVersionString must be numeric (major.minor.patch).
short_version="$(echo "$version" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"
sed -e "s/@VERSION@/$version/g" -e "s/@SHORT_VERSION@/$short_version/g" "$repo/packaging/macos/Info.plist" > "$contents/Info.plist"
printf 'APPL????' > "$contents/PkgInfo"

# Icon: the 256 px PNG of the Linux packaging, scaled into an iconset (no upscaling past the source size).
iconset="$work/sharplight-studio.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256; do
    sips -z "$size" "$size" "$repo/packaging/linux/sharplight-studio.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    if [ "$double" -le 256 ]; then
        sips -z "$double" "$double" "$repo/packaging/linux/sharplight-studio.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
    fi
done
iconutil -c icns "$iconset" -o "$contents/Resources/sharplight-studio.icns"

# Apple Silicon only runs signed code: sign ad hoc (no identity) until Developer ID signing is set up (docs/macos.md).
codesign --force --deep --sign - --timestamp=none "$app"
codesign --verify --deep --strict "$app"

zip="$out/Sharplight-Studio-${version}-macos-arm64.zip"
rm -rf "$out/Sharplight Studio.app" "$zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$zip"
ditto "$app" "$out/Sharplight Studio.app"
echo "$zip"
