#!/bin/bash
. <env.sh>
for r in $(seq 1 ${ROUNDS:-2}); do for b in "$@"; do printf "r$r %-14s " $b; SECONDS_RUN=${SEC:-4} $WORK/bin/$b; done; done
