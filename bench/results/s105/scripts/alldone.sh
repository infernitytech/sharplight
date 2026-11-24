#!/bin/bash
# exits 0 when no queued perf-r45 job is alive
pgrep -f 'sat_r2.sh|open.sh|redo_dec16.sh|ab_serve.sh|ab_skip.sh|vprof.sh|rl_all.sh|scripts/rl.sh|scripts/sat.sh|scripts/greedy.sh|scripts/tests.sh|s54_vllm_decode_profile|s16_serve.sh' > /dev/null && exit 1
pgrep -f 'kvq8 |for m in qwen llama gptoss|for p in cuda vulkan' >/dev/null && exit 1
exit 0
