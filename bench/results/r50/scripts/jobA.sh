#!/bin/bash
# GPU job A (baseline = develop + r48): BatchBench host/GPU per step, first runs; serving profiles; vLLM gaps. Run under the bench lock.
. <path>/env.sh
export HOME=<home>/test-home NUGET_PACKAGES=<path>/packages DOTNET_CLI_HOME=<home> SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
BB=${BB:-<path>/BatchBench}; O=${O:-<path>/base}; mkdir -p $O/steps $O/serve
G=<path>/gemma-4-31B-it-Q4_K_M.gguf; Q=<path>/gemma-4-31B-it-qat-w4a16-ct; L=<path>/Llama-3.1-8B-Instruct
b() { local tag=$1; shift; uptime > $O/steps/$tag.load; timeout 1500 $BB "$@" --provider cuda --dump-steps $O/steps/$tag.csv > $O/steps/$tag.txt 2>&1; uptime >> $O/steps/$tag.load; }
[ "${SKIP_PROBE:-0}" = 1 ] || {
b probe_llama $L --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 4
b probe_gemma $G --slots 64 --ctx 64 --depth 16 --batch 2048 --sizes 64 --prepare-probe --prepare-rounds 4
DOTNET_ROLL_FORWARD=Major timeout 900 <path>/dotnet-trace collect --profile dotnet-sampled-thread-time --format Speedscope -o $O/steps/probe_llama.nettrace -- $BB $L --provider cuda --slots 256 --ctx 64 --depth 16 --batch 2048 --sizes 256 --prepare-probe --prepare-rounds 8 > $O/steps/probe_llama_trace.txt 2>&1
}
[ "${SKIP_BB:-0}" = 1 ] || {
b gemma_q4 $G --slots 16 --ctx 1024 --depth 512 --batch 2048 --modes greedy,download --reps 30 --sizes 1,16
b llama_mixed $L --slots 32 --ctx 512 --depth 256 --batch 2048 --modes greedy --reps 10 --mixed 16+64,16+98,16+160,13+240,14+100,15+200
b llama_dec $L --slots 64 --ctx 512 --depth 128 --batch 2048 --modes greedy --reps 20 --sizes 1,8,16,32,64
}
S=<path>/servprof.sh
[ "${SKIP_SERVE:-0}" = 1 ] || {
for c in 1 16; do PER=$([ $c = 1 ] && echo 8 || echo 4) SRV=${SRV:-<path>/sharplight} $S $O/serve sl gemma_q4 $G $c; done
for c in 1 16; do PER=$([ $c = 1 ] && echo 8 || echo 4) SRV=${SRV:-<path>/sharplight} $S $O/serve sl gemma_qat $Q $c; done
}
[ "${VLLM:-1}" = 1 ] && for c in 1 16; do PER=$([ $c = 1 ] && echo 8 || echo 4) $S $O/serve vllm gemma_qat $Q $c --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"audio":0}'; done
echo jobA-done
