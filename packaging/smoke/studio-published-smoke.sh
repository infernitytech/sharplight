#!/usr/bin/env bash
# Smoke test of the Studio build users get: downloads the newest package of the unstable channel from the update feed exactly as the in-app
# updater finds it (releases.<rid>-<channel>.json, then the installer next to it), starts it headless with portable storage in a temporary
# folder and runs `--smoke` (services, devices, a feed install of a tiny package, one inference on the CPU, a JSON report).
#
#   studio-published-smoke.sh <rid> <out-dir>
#
# rid: linux-x64 | win-x64 | osx-arm64. Environment:
#   STUDIO_UPDATE_URL   feed folder (default https://download.sharplight.dev/studio)
#   SMOKE_CHANNEL       unstable (default) or stable
#   EXPECTED_SHA        commit that was just published; the downloaded version must be built from it (or from a newer commit on it)
#   SMOKE_CUDA          auto (default): also run `--smoke --smoke-provider cuda` where nvidia-smi finds a GPU (linux-x64, win-x64; skipped with a
#                       note otherwise), 0: never, 1: a missing GPU fails the run
#   GITHUB_STEP_SUMMARY optional: a Markdown summary is appended
# Writes <out-dir>/report.json (report-cuda.json for the CUDA run), <out-dir>/storage.txt, <out-dir>/logs/ (the Studio log folder) and <out-dir>/run.txt; exit 0 = passed.
# Needs bash, curl, jq; Linux also xvfb-run; Windows 7-Zip or PowerShell; macOS ditto (all present on the GitHub runners).
set -euo pipefail

rid="${1:?usage: $0 <rid> <out-dir>}"
out="${2:?usage: $0 <rid> <out-dir>}"
base="${STUDIO_UPDATE_URL:-https://download.sharplight.dev/studio}"
channel="${SMOKE_CHANNEL:-unstable}"
feed="$base/releases.$rid-$channel.json"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

summary() {
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        echo "$1" >> "$GITHUB_STEP_SUMMARY"
    fi
}

fail() {
    echo "::error title=Studio smoke ($rid)::$1"
    summary "**$rid: FAILED** - $1"
    exit 1
}

# 1. The newest full release of the channel, as the updater reads it.
echo "Reading $feed"
curl -fsSL --retry 3 "$feed" -o "$work/feed.json" || fail "cannot read the update feed $feed"
# Compare the numeric core of the version (0.3.45), not the string: 0.3.9 < 0.3.10.
version="$(jq -r '[.Assets[] | select(.Type == "Full")] | max_by(.Version | split("-")[0] | split(".") | map(tonumber)) | .Version' "$work/feed.json")"
[ -n "$version" ] && [ "$version" != "null" ] || fail "the feed lists no full release"
echo "Newest $channel release: $version"

# 2. It must come from the commit that was just published (or a newer one on top of it).
if [ -n "${EXPECTED_SHA:-}" ]; then
    built="${version##*.g}"
    case "$EXPECTED_SHA" in
        "$built"*) echo "Version $version is built from the published commit $EXPECTED_SHA." ;;
        *)
            if git merge-base --is-ancestor "$EXPECTED_SHA" "$built" 2> /dev/null; then
                echo "::notice::The feed already has a newer build ($version) than the published commit $EXPECTED_SHA; testing it."
            else
                fail "the feed's newest version $version is not built from the published commit $EXPECTED_SHA"
            fi
            ;;
    esac
fi

# 3. Download the package of this OS; every installer of the OS must be there (a HEAD request for the others).
case "$rid" in
    linux-x64) names=("SharplightStudio-$rid-$channel.AppImage"); run="${names[0]}" ;;
    win-x64) names=("SharplightStudio-$rid-$channel-Setup.exe" "SharplightStudio-$rid-$channel-Portable.zip"); run="${names[1]}" ;;
    osx-arm64) names=("SharplightStudio-$rid-$channel-Setup.pkg" "SharplightStudio-$rid-$channel-Portable.zip"); run="${names[1]}" ;;
    *) fail "unknown runtime identifier $rid" ;;
esac
for name in "${names[@]}"; do
    if [ "$name" != "$run" ]; then
        curl -fsSIL --retry 3 "$base/$name" -o /dev/null || fail "$name is not published"
    fi
done
echo "Downloading $run"
curl -fsSL --retry 3 "$base/$run" -o "$work/$run" || fail "cannot download $run"

# 4. Unpack into a fresh folder. Portable storage: the marker file next to the package (Linux: next to the AppImage; Windows: next to
#    Update.exe). macOS: the .app bundle is unsigned and not modified, so storage goes to SHARPLIGHT_HOME in the temporary folder.
app="$work/app"
mkdir -p "$app"
unset SHARPLIGHT_HOME || true
case "$rid" in
    linux-x64)
        cp "$work/$run" "$app/$run"
        chmod +x "$app/$run"
        : > "$app/sharplight.portable"
        export APPIMAGE_EXTRACT_AND_RUN=1
        studio=("$app/$run")
        runner=(xvfb-run -a --server-args="-screen 0 1280x800x24")
        ;;
    win-x64)
        if command -v 7z > /dev/null; then
            7z x -y "-o$(cygpath -m "$app")" "$(cygpath -m "$work/$run")" > /dev/null
        else
            powershell -NoProfile -Command "Expand-Archive -Path '$(cygpath -w "$work/$run")' -DestinationPath '$(cygpath -w "$app")'"
        fi
        update="$(find "$app" -maxdepth 3 -name Update.exe | head -n 1)"
        exe="$(find "$app" -maxdepth 4 -name sharplight-studio.exe | head -n 1)"
        [ -n "$exe" ] || fail "sharplight-studio.exe is not in $run"
        marker_dir="$(dirname "${update:-$exe}")"
        : > "$marker_dir/sharplight.portable"
        studio=("$exe")
        runner=()
        ;;
    osx-arm64)
        ditto -x -k "$work/$run" "$app"
        exe="$(find "$app" -path '*.app/Contents/MacOS/sharplight-studio' | head -n 1)"
        [ -n "$exe" ] || fail "the .app bundle is not in $run"
        bundle="${exe%/Contents/MacOS/*}"
        # Apple Silicon only runs signed code: an unsigned bundle gets an ad-hoc signature (no identity), as the packaging does.
        if ! codesign --verify --deep "$bundle" 2> /dev/null; then
            codesign --force --deep --sign - "$bundle"
        fi
        xattr -cr "$bundle" || true
        mkdir -p "$work/home"
        export SHARPLIGHT_HOME="$work/home"
        studio=("$exe")
        runner=()
        ;;
esac

# 5. Where will it store things? (--print-storage is the resolved layout of the build under test.)
${runner[@]+"${runner[@]}"} "${studio[@]}" --print-storage > "$out/storage.txt" 2>&1 || fail "--print-storage failed: $(head -c 600 "$out/storage.txt")"
cat "$out/storage.txt"
if [ "$rid" != "osx-arm64" ] && ! grep -qi '^Mode:.*portable' "$out/storage.txt"; then
    fail "the build did not select portable storage: $(grep '^Mode:' "$out/storage.txt")"
fi

# 6. The smoke run.
status=0
report_arg="$out/report.json"
if [ "$rid" = "win-x64" ]; then
    report_arg="$(cygpath -m "$report_arg")"
fi
${runner[@]+"${runner[@]}"} "${studio[@]}" --smoke --smoke-report "$report_arg" > "$out/run.txt" 2>&1 || status=$?
echo "Studio --smoke exited with $status"
tail -n 40 "$out/run.txt" || true

logs="$(sed -n 's/^Logs: *//p' "$out/storage.txt" | head -n 1)"
if [ "$rid" = "win-x64" ] && command -v cygpath > /dev/null && [ -n "$logs" ]; then
    logs="$(cygpath -u "$logs")"
fi
mkdir -p "$out/logs"
if [ -n "$logs" ] && [ -d "$logs" ]; then
    cp -R "$logs/." "$out/logs/"
fi

[ -f "$out/report.json" ] || fail "Studio exited with $status and wrote no report; log: $(tail -n 15 "$out/logs/studio.log" 2> /dev/null | tr '\n' ' ')"
passed="$(jq -r '.Passed' "$out/report.json")"
reported="$(jq -r '.Version' "$out/report.json")"
summary "### $rid ($channel) $version"
summary "$(jq -r '"- devices: " + (.Devices | join("; ")) + "\n" + ([.Steps[] | "- " + (if .Passed then "pass" else "FAIL" end) + " " + .Name + " (" + (.Seconds | tostring) + " s): " + .Detail] | join("\n"))' "$out/report.json")"
if [ "$reported" != "${version}" ]; then
    fail "the build reports version $reported but the feed says $version"
fi
if [ "$passed" != "true" ] || [ "$status" -ne 0 ]; then
    fail "smoke run failed (exit $status): $(jq -r '.Error' "$out/report.json")"
fi
# 7. The same run on CUDA, where the machine has an NVIDIA GPU (the builds of linux-x64 and win-x64 carry the CUDA provider and NVRTC). The GitHub
#    runners have none: the step is then skipped with a note, not failed. SMOKE_CUDA=0 skips it, SMOKE_CUDA=1 makes a missing GPU a failure.
cuda="${SMOKE_CUDA:-auto}"
case "$rid" in
    linux-x64|win-x64) ;;
    *) cuda=0 ;;
esac
if [ "$cuda" != "0" ]; then
    if command -v nvidia-smi > /dev/null && nvidia-smi -L > "$out/nvidia-smi.txt" 2>&1; then
        echo "NVIDIA GPU found ($(head -n 1 "$out/nvidia-smi.txt")): running the smoke test on CUDA"
        cuda_report="$out/report-cuda.json"
        report_arg="$cuda_report"
        if [ "$rid" = "win-x64" ]; then
            report_arg="$(cygpath -m "$report_arg")"
        fi
        cuda_status=0
        ${runner[@]+"${runner[@]}"} "${studio[@]}" --smoke --smoke-provider cuda --smoke-report "$report_arg" > "$out/run-cuda.txt" 2>&1 || cuda_status=$?
        echo "Studio --smoke --smoke-provider cuda exited with $cuda_status"
        tail -n 20 "$out/run-cuda.txt" || true
        [ -f "$cuda_report" ] || fail "the CUDA smoke run exited with $cuda_status and wrote no report"
        summary "### $rid CUDA"
        summary "$(jq -r '"- devices: " + (.Devices | join("; ")) + "\n" + ([.Steps[] | "- " + (if .Passed then "pass" else "FAIL" end) + " " + .Name + " (" + (.Seconds | tostring) + " s): " + .Detail] | join("\n"))' "$cuda_report")"
        if [ "$(jq -r '.Passed' "$cuda_report")" != "true" ] || [ "$cuda_status" -ne 0 ]; then
            fail "the CUDA smoke run failed (exit $cuda_status): $(jq -r '.Error' "$cuda_report")"
        fi
        echo "Studio smoke ($rid, CUDA) passed"
    elif [ "$cuda" = "1" ]; then
        fail "SMOKE_CUDA=1 but this machine has no NVIDIA GPU (nvidia-smi)"
    else
        echo "::notice title=Studio smoke ($rid)::CUDA smoke skipped: this runner has no NVIDIA GPU (the CPU smoke run above passed)"
        summary "- CUDA smoke skipped: no NVIDIA GPU on this runner"
    fi
fi

echo "Studio smoke ($rid) passed: $version"
