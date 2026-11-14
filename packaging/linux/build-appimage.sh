#!/usr/bin/env bash
# Builds the Sharplight Studio AppImage: a self-contained linux-x64 publish wrapped with appimagetool.
# Usage: packaging/linux/build-appimage.sh <version> [output directory]
# Also writes a .tar.gz of the same app folder for users who prefer to unpack it.
set -euo pipefail

version="${1:?version required}"
out="$(realpath -m "${2:-artifacts/studio}")"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

appdir="$work/AppDir"
app="$appdir/usr/lib/sharplight-studio"
mkdir -p "$app" "$appdir/usr/share/applications" "$appdir/usr/share/icons/hicolor/scalable/apps" \
    "$appdir/usr/share/icons/hicolor/256x256/apps" "$out"

dotnet publish "$repo/apps/Sharplight.Studio/Sharplight.Studio.csproj" -c Release -r linux-x64 --self-contained \
    -p:PublishReadyToRun=true -o "$app"

install -m 755 "$repo/packaging/linux/AppRun" "$appdir/AppRun"
install -m 644 "$repo/packaging/linux/sharplight-studio.desktop" "$appdir/sharplight-studio.desktop"
install -m 644 "$repo/packaging/linux/sharplight-studio.desktop" "$appdir/usr/share/applications/"
install -m 644 "$repo/packaging/linux/sharplight-studio.png" "$appdir/sharplight-studio.png"
install -m 644 "$repo/packaging/linux/sharplight-studio.png" "$appdir/usr/share/icons/hicolor/256x256/apps/"
install -m 644 "$repo/packaging/linux/sharplight-studio.svg" "$appdir/usr/share/icons/hicolor/scalable/apps/"
ln -s sharplight-studio.png "$appdir/.DirIcon"
chmod -R a+rX "$appdir"

tool="${APPIMAGETOOL:-}"
if [ -z "$tool" ]; then
    tool="$work/appimagetool"
    curl -fsSL -o "$tool" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$tool"
fi

name="Sharplight_Studio-${version}-x86_64"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$out/$name.AppImage"
tar -C "$appdir/usr/lib" -czf "$out/sharplight-studio-${version}-linux-x64.tar.gz" sharplight-studio
echo "$out/$name.AppImage"
