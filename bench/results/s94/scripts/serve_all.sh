#!/bin/bash
H=<out>/hold.sh; A=<out>/serve_ab.sh; S=<out>/serve; mkdir -p $S
ARRIVAL=closed $H $A $S/closed "new vllm new" qwen > $S/closed_qwen.log 2>&1
ARRIVAL=closed $H $A $S/closed_g "new vllm new" gptoss > $S/closed_gptoss.log 2>&1
echo all-done
