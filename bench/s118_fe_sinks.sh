#!/bin/bash
# Section 118: causal `fe` with per-head sinks on Vulkan (gpt-oss) against the `fa` fallback (SHARPLIGHT_ATTN_FE_SINKS=0).
#   MODEL=<gpt-oss-20b HF dir> bench/s118_fe_sinks.sh [prefill|profile]
# prefill: `--bench N` (one N-row chunk) for both settings, interleaved, REPS times each.
# profile: per-node GPU times of one 2048-row prefill (SHARPLIGHT_VK_PROFILE=2), summed by kernel class.
SL=samples/Sharplight.Cli/bin/Release/net11.0/sharplight
MODEL=${MODEL:?set MODEL}
REPS=${REPS:-3}
what=${1:-prefill}
case $what in
  prefill)
    for t in 512 2048 8192; do for r in $(seq "$REPS"); do for s in 1 0; do
      res=$(SHARPLIGHT_ATTN_FE_SINKS=$s $SL "$MODEL" --provider vulkan --bench $t --ctx $((t + 256)) --batch $t 2>&1 | grep '^prefill' | cut -d'|' -f1)
      echo "pp$t sinks_on_fe=$s rep$r | $res"
    done; done; done ;;
  profile)
    for s in 1 0; do
      echo "== sinks_on_fe=$s"
      SHARPLIGHT_ATTN_FE_SINKS=$s SHARPLIGHT_VK_PROFILE=2 SHARPLIGHT_PROFILE_TOP=2000 $SL "$MODEL" --provider vulkan --bench 2048 --ctx 2304 --batch 2048 2>&1 \
        | awk '/prefill T=/{p=1} /decode x/{p=0} p' | grep -E "GPU time|attention"  \
        | sed -E 's/layers\.[0-9]+\.//' | awk '/GPU time/{print; next} {a[$NF]+=$1; n[$NF]++} END{for (k in a) printf "%s: %.3f ms over %d launches\n", k, a[k], n[k]}'
    done ;;
esac
