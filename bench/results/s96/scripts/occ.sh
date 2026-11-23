#!/bin/bash
. <env.sh>
for i in 1 2; do for cfg in "rk_0_32" "rk_0_32 SMEM_KB=34" "rk_0_32_s8 SMEM_KB=34" "rk_0_64 SMEM_KB=34" "rk_0_32 SMEM_KB=24"; do set -- $cfg; b=$1; shift; printf "%-24s " "$cfg"; env SECONDS_RUN=3 "$@" $WORK/bin/$b; done; done
