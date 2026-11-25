#!/bin/bash
. $OUT/env.sh
O=$OUT/exp1; mkdir -p $O; D=$WORK/exp1; mkdir -p $D
export XDG_CACHE_HOME=$WORK/cache
run() { local lab=$1; shift; local t0=$(date +%s)
  env "$@" $PPL $M27 --provider cuda --text $TEXT/wikitext2_test.txt --seq 1024 --chunk 512 --tokens 2048 --dump $D/$lab.f16 > $O/$lab.txt 2>&1
  echo "$lab [$*] $(($(date +%s)-t0))s $(md5sum < $D/$lab.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*$' $O/$lab.txt | tail -1)"; }
run wg0 SHARPLIGHT_FP8_WG=0
run wg1a SHARPLIGHT_FP8_WG=1
run wg1b SHARPLIGHT_FP8_WG=1
run nopdl_a SHARPLIGHT_FP8_WG=1 SHARPLIGHT_CUDA_PDL_TRIGGER=0
run nopdl_b SHARPLIGHT_FP8_WG=1 SHARPLIGHT_CUDA_PDL_TRIGGER=0
run nopdl_c SHARPLIGHT_FP8_WG=1 SHARPLIGHT_CUDA_PDL_TRIGGER=0
