#!/bin/bash
# GPU job C: the alternating +48 ms host stall of Gemma 4 31B 1-row BatchBench steps; Gemma steps base vs branch at a fitting depth.
. <path>/env.sh
export HOME=<home>/test-home NUGET_PACKAGES=<path>/packages DOTNET_CLI_HOME=<home> SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
G=<path>/gemma-4-31B-it-Q4_K_M.gguf; O=<path>/c; mkdir -p $O
b() { local bb=$1 tag=$2; shift 2; uptime > $O/$tag.load; timeout 1500 $bb "$@" --provider cuda --dump-steps $O/$tag.csv > $O/$tag.txt 2>&1; uptime >> $O/$tag.load; }
BB=<path>/BatchBench; BR=<path>/BatchBench
b $BB stall_d512 $G --slots 4 --ctx 1024 --depth 512 --batch 2048 --modes greedy --reps 20 --sizes 1
b $BB stall_d128 $G --slots 4 --ctx 1024 --depth 128 --batch 2048 --modes greedy --reps 20 --sizes 1
SHARPLIGHT_SEGMENT_TIMING=1 b $BB stall_seg $G --slots 4 --ctx 1024 --depth 512 --batch 2048 --modes greedy --reps 20 --sizes 1
DOTNET_ROLL_FORWARD=Major timeout 900 <path>/dotnet-trace collect --profile dotnet-sampled-thread-time --format Speedscope -o $O/stall.nettrace -- $BB $G --provider cuda --slots 4 --ctx 1024 --depth 512 --batch 2048 --modes greedy --reps 200 --sizes 1 > $O/stall_trace.txt 2>&1
for r in 1 2; do
b $BB gemma_base_$r $G --slots 16 --ctx 640 --depth 448 --batch 2048 --modes greedy,download --reps 30 --sizes 1,16
b $BR gemma_br_$r $G --slots 16 --ctx 640 --depth 448 --batch 2048 --modes greedy,download --reps 30 --sizes 1,16
done
echo jobC-done
