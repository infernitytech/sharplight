#!/bin/bash
# usage: build.sh <tree> [projects...]
. $TOOLS/env.sh
T=$1; shift
cd $T
for q in ${@:-samples/Sharplight.Cli bench/BatchBench bench/PplBench bench/onnx/OnnxBench}; do
  echo "== $q"
  nice -n 15 ionice -c3 dotnet build $q -c Release -m:4 -nologo -v q 2>&1 | grep -E " error |Elapsed" | tail -5
done
echo build-done
