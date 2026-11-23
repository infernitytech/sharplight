. <dev-env.sh>
export TREE=<tree> BASE=<base-tree> OUT=<out> WORK=<work>
export M27=~/models/hf/Qwen3.8-27B-FP8 M8=~/models/hf/Qwen3-8B-FP8 MOSS=~/models/hf/gpt-oss-20b
export PPL=$TREE/bench/PplBench/bin/Release/net11.0/PplBench PPLB=$BASE/bench/PplBench/bin/Release/net11.0/PplBench
export SASS_EXTRA='#define SL_NVRTC_SASS 1|void quantize_fp8(|void decode_combine('
