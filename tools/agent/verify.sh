#!/usr/bin/env bash
# usage: verify.sh <tag> [suite...]
# Builds develop of $SL_REPO in the detached worktree $SL_VERIFY under the verify lock: workflow YAML check, style checks,
# full Release build, then the given suites through run-suites.sh (GPU suites on the indices in GPUS, default "0").
# Prints "RESULT-<tag> done <commit>" on success; logs go to $SL_WORK/logs/verify-<tag>.
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
tag=$1; shift
exec 9>"$SL_STATE/.sl-verify.lock"; flock 9
cd "$SL_VERIFY" && git checkout -q --detach develop
L=$SL_WORK/logs/verify-$tag; rm -rf "$L"; mkdir -p "$L"
python3 -c "import yaml,glob;[yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]" > "$L/yaml.log" 2>&1 \
    || { echo "RESULT-$tag yamlfail"; exit 1; }
{ python3 tools/check-one-type-per-file.py > "$L/style.log" 2>&1 && python3 tools/check-single-line-scopes.py >> "$L/style.log" 2>&1; } \
    || { echo "RESULT-$tag stylefail"; tail -5 "$L/style.log"; exit 1; }
# -nodeReuse:false and 9>&-: a reused MSBuild node or compiler server must not inherit (and keep holding) the verify lock.
nice -n 15 ionice -c3 dotnet build Sharplight.slnx -c Release -m:4 -nodeReuse:false -p:UseSharedCompilation=false 9>&- > "$L/build.log" 2>&1 \
    || { echo "RESULT-$tag buildfail"; grep -E ' error ' "$L/build.log" | head; exit 1; }
if [ $# -gt 0 ]; then
    SUITE_JOBS=${SUITE_JOBS:-1} SUITE_GPUS="${GPUS:-0}" "$here/run-suites.sh" "$SL_VERIFY" "$L" "$@" 9>&- > "$L/summary.txt" 2>&1
fi
cat "$L/summary.txt" 2>/dev/null
echo "RESULT-$tag done $(git rev-parse --short HEAD)"
