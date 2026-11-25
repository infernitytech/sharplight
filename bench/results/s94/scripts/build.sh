#!/bin/bash
# usage: build.sh <tree> [projects...]
. <dev-env.sh>
T=$1; shift
cd $T
for p in "${@:-samples/Sharplight.Cli bench/BatchBench bench/PplBench}"; do
  for q in $p; do
    echo "== $q"
    nice -n 15 ionice -c3 dotnet build $q -c Release -m:4 -nologo -v q 2>&1 | grep -E "error|Warn|Elapsed" | grep -v "warning" | tail -5
  done
done
echo build-done
