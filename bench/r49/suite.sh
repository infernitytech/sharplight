#!/bin/bash
# usage: suite.sh <worktree> <logdir> <Suite> [provider]  -- one class per process, each under the bench lock (GPU suites)
export TMPDIR=${TMPDIR:-/tmp}
wt=$1; logs=$2; s=$3; prov=${4:-}
mkdir -p "$logs/$s${prov:+-$prov}"; cd "$wt" || exit 1
export SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
[ -n "$prov" ] && export SHARPLIGHT_TEST_PROVIDER=$prov
strip() { sed 's/\x1b\[[0-9;]*m//g'; }
sum=$logs/$s${prov:+-$prov}.log; : > "$sum"
classes=$(timeout 300 dotnet run --project tests/Sharplight.$s.Tests -c Release --no-build -- -list classes 2>&1 | strip | grep -E '^[A-Za-z_][A-Za-z0-9_.]*$')
run_class() { # one class, no lock (the caller holds it)
  local c=$1 l=$logs/$s${prov:+-$prov}/${1##*.}.log rc
  timeout 3600 nice -n 15 dotnet run --project tests/Sharplight.$s.Tests -c Release --no-build -- -class "$c" > "$l" 2>&1
  rc=$?; echo "$c exit=$rc $(strip < "$l" | grep -o 'Total: .*' | tail -1)" >> "$sum"
  [ $rc != 0 ] && strip < "$l" | grep -E '\[FAIL\]' | head -10 >> "$sum"
}
export -f run_class strip; export logs s prov sum
# GROUP classes per hold of the bench lock (each class still its own process)
set -- $classes
while [ $# -gt 0 ]; do
  grp=("${@:1:${GROUP:-8}}"); shift ${#grp[@]} 2>/dev/null || shift $#
  flock ${SL_BENCH_LOCK:-/tmp/sl-bench.lock} bash -c 'for c in "$@"; do run_class "$c"; done' _ "${grp[@]}"
done
echo "DONE" >> "$sum"
