#!/bin/bash
# tests.sh: suites one class per process (GPU tests inside one bench-lock hold). SUITES / FILES / SHARPLIGHT_TEST_PROVIDER select.
. $(dirname "$0")/env.sh
O=$OUT/tests/${TAG:-cuda}; mkdir -p $O; cd $TREE
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_TEST_PROVIDER=${SHARPLIGHT_TEST_PROVIDER:-cuda} HOME=~/test-home
export XDG_CACHE_HOME=$WORK/cache_t
echo "# hold $(date +%T)"
for p in ${SUITES:-Sharplight.Cuda.Tests Sharplight.Serving.Tests}; do
  for f in ${FILES:-$(grep -rlE "\[(Fact|Theory)" tests/$p --include=*.cs)}; do
    ns=$(grep -m1 -oP '^namespace \K[\w.]+' $f); for c in $(grep -oP '(public|internal) (sealed |static |partial )*class \K\w+' $f); do
      nice -n 15 dotnet run --project tests/$p -c Release --no-build -- -class "$ns.$c" 9>&- > $O/${p}_$c.txt 2>&1
      echo "$p $c rc=$? $(grep -oE 'Total: [0-9]+, Errors: [0-9]+, Failed: [0-9]+, Skipped: [0-9]+' $O/${p}_$c.txt | tail -1)"
    done
  done
done
echo "# release $(date +%T)"
