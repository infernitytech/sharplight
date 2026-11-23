#!/bin/bash
# greedy text and PplBench, base vs new, one hold
. /workspace/work/perf-r40/env.sh
cd /workspace/work/perf-r40
exec 9>/root/.sl-verify.lock; flock 9
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
./greedy.sh "gpt-oss-20b:$MOSS" "Qwen3-8B-FP8:$M8" "Qwen2.5-1.5B:/root/models-local/hf/Qwen2.5-1.5B-Instruct" 9>&-
for m in "$MOSS gpt-oss-20b" "$M8 Qwen3-8B-FP8"; do set -- $m
  ./ppl.sh base $1 $2 1 9>&-; ./ppl.sh new $1 $2 2 9>&-
done
echo "# release $(date +%T)"
