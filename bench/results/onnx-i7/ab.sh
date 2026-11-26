#!/bin/bash
# usage: ab.sh <manifest> <provider> <rounds> [extra args]   -> prints gpuMin/min per engine per round
eval "$(~/.dotnetup/dotnetup env script --shell bash --dotnet --dotnetup)" 2>/dev/null
export SHARPLIGHT_ONNX_MODELS=~/models/onnx SHARPLIGHT_VK_DEVICES=0
m=$1; p=$2; r=$3; shift 3
NEW=~/.multiclaude/workspaces/sharplight~multi/slot-1/sharplight/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll
BASE=/tmp/sl-base/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll
for i in $(seq $r); do for e in base new; do d=$BASE; [ $e = new ] && d=$NEW
dotnet $d $m $p --runs 300 --budget-s 8 "$@" 2>&1 | python3 -c "
import json,sys;l=sys.stdin.read();j=json.loads(l[l.index('{'):])
print('$e', j.get('stage'), j.get('parityOk'), 'worstRms=%.2e'%(j.get('worstRms') or 0), 'min=%s med=%s gpuMin=%s gpuMed=%s'%(j.get('minMs'),j.get('medMs'),j.get('gpuMinMs'),j.get('gpuMedMs')), j.get('error') or '')"
done; done
