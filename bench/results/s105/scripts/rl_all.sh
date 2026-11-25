#!/bin/bash
R=<path>
for m in ${MODELS:-qwen gptoss llama}; do
  [ "${SKIPA:-}" = $m ] || CTX=2560 $R $m A "1+0,16+0,64+0,128+0,0+2048,16+240"
  HOLD=${HOLDB:-1500} $R $m B "256+0,256+256x7,128+240x8"
done
