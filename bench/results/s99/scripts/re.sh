#!/bin/bash
# read_energy layouts: cp.async vs register path, sector halves together vs split; one hold, 2 alternated rounds
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for r in 1 2; do for b in re_0_0_32 re_0_1_32 re_3_0_32 re_3_1_32 re_3_0_64 re_3_1_64 re_3_0_128 re_3_1_128; do
  echo -n "$r $b: "; SECONDS_RUN=3 $WORK/re/$b 9>&-
done; done
echo "# release $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
