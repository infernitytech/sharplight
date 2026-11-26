#!/usr/bin/env bash
# Runs Sharplight test suites, at most SUITE_JOBS (default 3) at a time, niced; GPU suites pinned round-robin over SUITE_GPUS (default "0").
# Usage: run-suites.sh <repo> <logdir> [suite...]
# GPU suites run ONE CLASS PER PROCESS (a whole-suite Vulkan/ImageGeneration process reaches ~28 GB RSS and was OOM-killed, exit 137,
# under a 62 GB memory limit). SUITE_PER_CLASS=0 runs every suite as one process; =all runs every suite per class.
# Suite names are test project names without the "Sharplight." prefix and with their ".Tests" suffix (Core.Tests, Vulkan.Tests).
# Each <logdir>/<suite>.log ends with a summed "Total: N, Errors: E, Failed: F, Skipped: S" line and "exit=X" (X = worst class exit).
set -u
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
repo=$1; logs=$2; shift 2
mkdir -p "$logs"
suites=("$@")
if [ ${#suites[@]} -eq 0 ]; then suites=($(ls "$repo/tests")); fi
gpu_suites="Vulkan Families Vision Serving Embeddings Studio Speech TextToSpeech AudioModels ImageGeneration Video Hosting Import.Onnx.Tests Import.Onnx.Conformance Extensions.AI"
gpus=(${SUITE_GPUS:-0})
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

run_suite() { # <suite> <device> <per-class 0|1>
  local s=$1 dev=$2 per=$3 log="$logs/$1.log"
  local env=(HOME=${TEST_HOME:-$SL_TEST_HOME} NUGET_PACKAGES=${NUGET_PACKAGES:-$HOME/.nuget/packages} DOTNET_CLI_HOME=$HOME SHARPLIGHT_VK_DEVICES=$dev)
  cd "$repo" || return
  if [ "$per" = 0 ]; then
    env "${env[@]}" timeout 7200 nice -n 15 ionice -c3 dotnet run --project tests/Sharplight.$s -c Release --no-build > "$log" 2>&1
    echo "exit=$?" >> "$log"; return
  fi
  local classes c rc worst=0 t=0 e=0 f=0 k=0 line
  classes=$(env "${env[@]}" timeout 300 dotnet run --project tests/Sharplight.$s -c Release --no-build -- -list classes 2>&1 | strip | grep -E '^[A-Za-z_][A-Za-z0-9_.]*$')
  : > "$log"; mkdir -p "$logs/$s"
  for c in $classes; do
    env "${env[@]}" timeout 3600 nice -n 15 ionice -c3 dotnet run --project tests/Sharplight.$s -c Release --no-build -- -class "$c" > "$logs/$s/${c##*.}.log" 2>&1
    rc=$?; [ $rc -gt $worst ] && worst=$rc
    line=$(strip < "$logs/$s/${c##*.}.log" | grep -o 'Total: .*' | tail -1)
    echo "$c exit=$rc $line" >> "$log"
    [ $rc != 0 ] && strip < "$logs/$s/${c##*.}.log" | grep -E '\[FAIL\]' | head -20 >> "$log"
    t=$((t + $(echo "$line" | grep -oP 'Total: \K[0-9]+' || echo 0)))
    e=$((e + $(echo "$line" | grep -oP 'Errors: \K[0-9]+' || echo 0)))
    f=$((f + $(echo "$line" | grep -oP 'Failed: \K[0-9]+' || echo 0)))
    k=$((k + $(echo "$line" | grep -oP 'Skipped: \K[0-9]+' || echo 0)))
  done
  [ -z "$classes" ] && worst=99 && echo "no classes listed (build missing?)" >> "$log"
  echo "Sharplight.$s  Total: $t, Errors: $e, Failed: $f, Skipped: $k (per class: $(echo "$classes" | grep -c .) processes)" >> "$log"
  echo "exit=$worst" >> "$log"
}

g=0; pids=()
for s in "${suites[@]}"; do
  s=${s#Sharplight.}; dev=""; per=0
  for x in $gpu_suites; do [[ "$s" == "$x"* ]] && dev=${gpus[$g]} && g=$(( (g+1) % ${#gpus[@]} )) && per=1; done
  case "${SUITE_PER_CLASS:-gpu}" in 0) per=0 ;; all) per=1 ;; esac
  while [ "$(jobs -rp | wc -l)" -ge "${SUITE_JOBS:-3}" ]; do wait -n; done
  run_suite "$s" "${dev:-0}" "$per" &
  pids+=($!)
done
wait "${pids[@]}"
for s in "${suites[@]}"; do s=${s#Sharplight.}; printf '%-28s %s %s\n' "$s" "$(grep -E 'Total: ' "$logs/$s.log" | tail -1 | strip | sed 's/^ *//')" "$(tail -1 "$logs/$s.log")"; done
