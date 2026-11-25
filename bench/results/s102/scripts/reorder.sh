#!/bin/bash
until [ -d /ab/gptoss_base_1 ]; do sleep 5; done
pkill -f 'serve_ab.sh /ab' ; sleep 1
pkill -TERM -f 's16_serve.sh sharplight gptoss'; sleep 15
pgrep -af 'sharplight serve|s16_serve' | head
rm -rf /ab/gptoss_base_1
. <path> export ROUNDS=1 CONCS="16 64"
cd <path>
./hold.sh ./serve_ab.sh $OUT/ab "base new vllm new base" llama gptoss > job8.log 2>&1
SUITES=Sharplight.Import.Onnx.Tests ./hold.sh ./tests.sh > job9.log 2>&1
./hold.sh ./serve_ab.sh $OUT/ab2 "new base" qwen > job10.log 2>&1
ARRIVAL=poisson ./hold.sh ./serve_ab.sh $OUT/abp "base new new base" llama > job11.log 2>&1
