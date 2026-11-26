#!/bin/bash
# GPU job B: branch build (templates, memoized shared conditions, graphs captured ahead) against the base; Gemma BatchBench for both.
. <path>/env.sh
export HOME=<home>/test-home NUGET_PACKAGES=<path>/packages DOTNET_CLI_HOME=<home> SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
G=<path>/gemma-4-31B-it-Q4_K_M.gguf; L=<path>/Llama-3.1-8B-Instruct
b() { local bb=$1 o=$2 tag=$3; shift 3; mkdir -p $o/steps; uptime > $o/steps/$tag.load; timeout 1500 $bb "$@" --provider cuda --dump-steps $o/steps/$tag.csv > $o/steps/$tag.txt 2>&1; uptime >> $o/steps/$tag.load; }
B=<path>/br; A=<path>/base
cd <path>/perf-r50 && timeout 900 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build -- -class Sharplight.Cuda.Tests.PreparedStepGraphTests > <path>/test_prepared.txt 2>&1
b <path>/BatchBench $B probe_llama $L --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 4
b <path>/BatchBench $B probe_gemma $G --slots 64 --ctx 64 --depth 16 --batch 2048 --sizes 64 --prepare-probe --prepare-rounds 1
DOTNET_ROLL_FORWARD=Major timeout 900 <path>/dotnet-trace collect --profile dotnet-sampled-thread-time --format Speedscope -o $B/steps/probe_llama.nettrace -- <path>/BatchBench $L --provider cuda --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 8 > $B/steps/probe_llama_trace.txt 2>&1
for r in 1 2; do
b <path>/BatchBench $A gemma_q4_$r $G --slots 16 --ctx 640 --depth 512 --batch 2048 --modes greedy,download --reps 30 --sizes 1,16
b <path>/BatchBench $B gemma_q4_$r $G --slots 16 --ctx 640 --depth 512 --batch 2048 --modes greedy,download --reps 30 --sizes 1,16
done
b <path>/BatchBench $B llama_mixed $L --slots 32 --ctx 512 --depth 256 --batch 2048 --modes greedy --reps 10 --mixed 16+64,16+98,16+160,13+240,14+100,15+200
b <path>/BatchBench $B llama_dec $L --slots 64 --ctx 512 --depth 128 --batch 2048 --modes greedy --reps 20 --sizes 1,8,16,32,64
S=<path>/servprof.sh
for c in 1 16; do PER=$([ $c = 1 ] && echo 8 || echo 4) NSYS_DELAY=14 SRV=<path>/sharplight $S $B/serve sl gemma_q4 $G $c; done
PER=4 NSYS_DELAY=14 SRV=<path>/sharplight $S $A/serve2 sl gemma_q4 $G 16
echo jobB-done
