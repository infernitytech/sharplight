#!/bin/bash
# usage: var.sh <log> <provider> <batch> <manifest-rel> "<label>|<tag>|<env...>" ...   (one lock hold, REPS interleaved reps)
. <env.sh>
export SHARPLIGHT_ONNX_MODELS=~/models/onnx SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$WORK/cache_onnx}
log=$1; prov=$2; b=$3; man=$4; shift 4
exec 9><gpu lock>; flock 9
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> $log
for rep in $(seq ${REPS:-2}); do for v in "$@"; do
  IFS='|' read -r label tag envs <<< "$v"
  case $tag in base) dll=$BASE/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll;;
    new) dll=$TREE/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll;;
    exp) dll=$EXP/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll;; *) dll=$tag;; esac
  r=$(env $envs dotnet $dll $SHARPLIGHT_ONNX_MODELS/ref/$man $prov --batch $b $EXTRA 9>&- | grep '^RESULT ' | cut -c8- | python3 -c "import json,sys; j=json.load(sys.stdin); print(j.get('stage'), j.get('minMs'), j.get('gpuMinMs'), j.get('parityOk'), j.get('worstRms'), str(j.get('error',''))[:200])")
  echo "$man b$b $prov [$label] $r" | tee -a $log
done; done
echo "# release $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> $log
