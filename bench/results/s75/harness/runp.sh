#!/bin/bash
. $ENV; export CUDA_VISIBLE_DEVICES=0 EVICT=read; B=$WORK/bin
for r in 1 2; do for c in GU128 DN128 GU64 DN64; do for v in b h; do
  echo "== r$r $c $v: $(timeout 300 $B/p${c}_$v $([ $r = 2 ] && echo x) | grep -E "check|T=" | awk '/check/{e=e" "$4} /T=/{t=t" "$(NF-3)} END{print "err" e " | ms" t}')"
done; done; done
