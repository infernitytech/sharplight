#!/bin/bash
# nsys_onnx.sh <manifest-rel> <batch> "<label>|<env>" ...  : nsys kernel traces of OnnxBench CUDA (graph nodes), one hold
. <env.sh>
export SHARPLIGHT_ONNX_MODELS=~/models/onnx SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=$WORK/cache_onnx
man=$1; b=$2; shift 2; n=$(basename $man .json)
dll=$TREE/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll
exec 9><gpu lock>; flock 9
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for v in "$@"; do IFS='|' read -r label envs <<< "$v"
  rm -f nsys/${n}_$label.nsys-rep
  env $envs nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node -o nsys/${n}_$label dotnet $dll $SHARPLIGHT_ONNX_MODELS/ref/$man cuda --batch $b 9>&- > nsys/${n}_$label.out 2>&1
  grep '^RESULT' nsys/${n}_$label.out | cut -c1-200
done
echo "# release $(date +%T)"
