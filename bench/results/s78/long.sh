#!/bin/bash
# long.sh <tree> <label> <outdir> [depth] [extra env...]: BatchBench 4-stream fed-back decode at depth (inside the lock); PROFILE=1 adds a profile
. <path>
T=$1; L=$2; O=$3; D=${4:-130048}; shift 4; mkdir -p $O
BB="dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll"
P=""; [ -n "$PROFILE" ] && P="--profile 4 --profile-mode greedy --profile-top 60"
env "$@" timeout 1500 nice -n 5 $BB ~/models/hf/gpt-oss-20b --provider cuda --slots 4 --ctx $((D + 512)) --depth $D --batch 2048 --sizes ${SIZES:-1,4} --modes greedy --reps ${REPS:-5} --feedback $P > $O/long_${L}_d$D.txt 2>&1
echo "long $L d$D: $(grep -E '^\| [0-9]+ ' $O/long_${L}_d$D.txt | awk -F'|' '{printf "%s:%s  ", $2, $5}')"
