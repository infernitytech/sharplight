#!/bin/bash
# After correct.sh: steps, serving, tests; each block in its own GPU hold.
. $OUT/env.sh
H=$OUT/hold.sh; A=$OUT/serve_ab.sh; S=$OUT/serve; mkdir -p $S $OUT/steps
$H $OUT/steps.sh > $OUT/steps/steps.txt 2>&1
ARRIVAL=closed $H $A $S/closed "base new vllm new base" qwen > $S/closed_qwen.log 2>&1
ARRIVAL=poisson $H $A $S/poisson "base new wfloor wfull full vllm" qwen > $S/poisson_qwen.log 2>&1
ARRIVAL=closed $H $A $S/closed_pol "wfloor wfull full new" qwen > $S/closed_pol_qwen.log 2>&1
ARRIVAL=closed $H $A $S/closed_g "base wfloor wfull vllm full" gptoss > $S/closed_gptoss.log 2>&1
ARRIVAL=poisson $H $A $S/poisson_g "base wfloor wfull full vllm" gptoss > $S/poisson_gptoss.log 2>&1
$H $OUT/tests_hold.sh $OUT/tests > $OUT/tests_summary.txt 2>&1
echo all-done
