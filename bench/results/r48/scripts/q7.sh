#!/bin/bash
# Queue 7: Gemma 4 31B Q4_K_M host time per step (decode-only steps, wall vs GPU, by mode), base and branch
B=~/sl-worktrees/perf-r48-base; G=~/models-local/feed1/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf
REPO=$B MODES=greedy,download,logits REPS=30 SLOTS=16 DEPTH=512 CTX=1024 ./sizes.sh gemma_base $G 1,16 --dump-steps ~/sl-worktrees/perf-r48/bench/results/r48/steps/gemma_base.csv
MODES=greedy,download,logits REPS=30 SLOTS=16 DEPTH=512 CTX=1024 ./sizes.sh gemma_br $G 1,16 --dump-steps ~/sl-worktrees/perf-r48/bench/results/r48/steps/gemma_br.csv
echo q7-done
