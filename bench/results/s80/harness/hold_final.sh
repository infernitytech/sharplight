#!/bin/bash
# Final hold: base (develop f28b1aa0) vs new (perf/llm-r27), alternated; vLLM in the same hold.
. <path>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
W=<path> O=$W/final; mkdir -p $O
N=~/sl-worktrees/perf-r27; B=~/sl-worktrees/r27-base
G=~/models-local/hf/gpt-oss-20b; Q=~/models-local/hf/Qwen3-8B-FP8
Q25=~/models-local/gguf/Qwen2.5-7B-Instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
if [ "$1" = A ]; then
echo "greedy $(date +%T)"
for t in base:$B new:$N; do IFS=: read l tr <<< "$t"; $W/greedy.sh $tr $l $O/greedy $G $Q; PROVIDERS=cuda $W/greedy.sh $tr $l $O/greedy $Q25; done
echo "steps $(date +%T)"
for r in 1 2; do for t in new:$N base:$B; do IFS=: read l tr <<< "$t"
  M=$G SIZES=1,4,16,64 $W/ab_steps.sh $O/steps$r ${l}_gptoss $tr --text
  M=$Q SIZES=1,16,64 $W/ab_steps.sh $O/steps$r ${l}_qwen $tr --text
  M=$G $W/ab_mixed.sh $O/mixed$r $l:$tr
done; done
echo "vulkan $(date +%T)"
for t in new:$N base:$B; do IFS=: read l tr <<< "$t"; PROV=vulkan M=$G SIZES=1,64 $W/ab_steps.sh $O/vulkan ${l}_gptoss $tr --text; done
fi
if [ "$1" = B ]; then
echo "serve $(date +%T)"
cd $N
export VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 ROUNDS=2 OUT_DIR=$O/serve
SHARPLIGHT_SERVER=$N/samples/Sharplight.Cli/bin/Release/net11.0/sharplight bench/s16_serve.sh sharplight gptoss_new $G "16 64" --provider cuda
SHARPLIGHT_SERVER=$B/samples/Sharplight.Cli/bin/Release/net11.0/sharplight bench/s16_serve.sh sharplight gptoss_base $G "16 64" --provider cuda
bench/s16_serve.sh vllm gptoss $G "16 64" --gpu-memory-utilization 0.6
fi
if [ "$1" = C ]; then
for r in 1 2; do for t in new:$N base:$B; do IFS=: read l tr <<< "$t"; M=$Q SIZES=1,16,64 $W/ab_steps.sh $O/stepsC$r ${l}_qwen $tr --text; M=$G SIZES=1,64 $W/ab_steps.sh $O/stepsC$r ${l}_gptoss $tr --text; done; done
cd $N; export VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 ROUNDS=2 OUT_DIR=$O/serve
SHARPLIGHT_SERVER=$N/samples/Sharplight.Cli/bin/Release/net11.0/sharplight bench/s16_serve.sh sharplight qwen_new $Q "16 64" --provider cuda
SHARPLIGHT_SERVER=$B/samples/Sharplight.Cli/bin/Release/net11.0/sharplight bench/s16_serve.sh sharplight qwen_base $Q "16 64" --provider cuda
bench/s16_serve.sh vllm qwen $Q "16 64" --gpu-memory-utilization 0.6
fi
echo "done $1 $(date +%T)"
