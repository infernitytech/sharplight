#!/bin/bash
# All serving runs of §92, each block in its own GPU hold.
S=$OUT/serve; mkdir -p $S
H=$WORK/hold.sh; A=$WORK/serve_ab.sh
ARRIVAL=closed $H $A $S/closed "base new vllm new base" qwen > $S/closed_qwen.log 2>&1
ARRIVAL=poisson $H $A $S/poisson "base new full trim vllm" qwen > $S/poisson_qwen.log 2>&1
ARRIVAL=closed $H $A $S/closed_pol "full trim" qwen > $S/closed_pol_qwen.log 2>&1
ARRIVAL=closed $H $A $S/closed_g "base vllm full" gptoss > $S/closed_gptoss.log 2>&1
ARRIVAL=poisson $H $A $S/poisson_g "base full trim vllm" gptoss > $S/poisson_gptoss.log 2>&1
echo all-done
