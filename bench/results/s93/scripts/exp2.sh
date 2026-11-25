#!/bin/bash
. $OUT/env.sh
O=$OUT/exp2; mkdir -p $O; D=$WORK/exp2; mkdir -p $D
export XDG_CACHE_HOME=$WORK/cache
run() { local lab=$1; shift; local t0=$(date +%s)
  env "$@" $PPL $M27 --provider cuda --text $TEXT/wikitext2_test.txt --seq 1024 --chunk 512 --tokens 2048 --dump $D/$lab.f16 > $O/$lab.txt 2>&1
  echo "$lab [$*] $(($(date +%s)-t0))s $(md5sum < $D/$lab.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*$' $O/$lab.txt | tail -1)"; }
run nodl_a SHARPLIGHT_FP8_WG=1 SHARPLIGHT_DISABLE_FEATURES=dependent_launches
run nodl_b SHARPLIGHT_FP8_WG=1 SHARPLIGHT_DISABLE_FEATURES=dependent_launches
run nodl_c SHARPLIGHT_FP8_WG=1 SHARPLIGHT_DISABLE_FEATURES=dependent_launches
run sync_a SHARPLIGHT_FP8_WG=1 SHARPLIGHT_CUDA_SYNC=1
