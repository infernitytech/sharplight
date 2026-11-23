#!/bin/bash
. <env.sh>
for r in 1 2; do for f in 4 5 6; do for shape in 4096_12288 12288_4096; do for bm in 16 32 64; do for lay in base new; do
  T=$bm; [ $bm = 16 ] && T=9
  printf "r$r %-5s " $lay; $WORK/kq/kq_${f}_${shape}_${bm}_$lay $T | tail -1
done; done; done; done; done
