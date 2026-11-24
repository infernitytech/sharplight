#!/bin/bash
. <path>
cd $OUT
./job2.sh
LL=~/models-local/hf/Llama-3.1-8B-Instruct Q8=~/models-local/hf/Qwen3-8B
./bb.sh tile 16+0,16+112,16+240,16+250,16+304,16+368,16+496,16+512 2 $LL $M8
./bb.sh tile 16+0,16+240,16+250,16+368,16+496 1 $Q8 $MOSS
