#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; T=$2; C=$3; path=$4
export XDG_CACHE_HOME=$WORK/$C
nsys profile -o $OUT/nsys/$tag --force-overwrite true -t cuda --cuda-graph-trace=node \
  dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 16+240 > $OUT/nsys/$tag.txt 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv -o $OUT/nsys/$tag $OUT/nsys/$tag.nsys-rep > /dev/null 2>&1
grep -E '^\| 16' $OUT/nsys/$tag.txt
