#!/bin/bash
# greedy text base vs new (CUDA), last-token logits of ~3K tokens, PplBench logits dumps x5 (Qwen3-8B-FP8, Qwen3.8-27B-FP8)
. $OUT/env.sh
export XDG_CACHE_HOME=$WORK/cache_c
O=$OUT/correct; mkdir -p $O; D=$WORK/ppl; mkdir -p $D
P40="Write a detailed essay about the role of water and power in modern history. Explain the history, give concrete examples and discuss open questions."
P1000="$(cat $OUT/p1000.txt)"
T=$TEXT/wikitext2_test.txt
for lab in base new; do [ $lab = base ] && W=$BASE || W=$TREE
  CLI=$W/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll
  for m in Qwen3-8B-FP8 Qwen3.8-27B-FP8; do for pn in p40 p1000; do [ $pn = p40 ] && P="$P40" || P="$P1000"
    timeout 900 dotnet $CLI ~/models/hf/$m --provider cuda -n 120 -p "$P" > $O/${m}_${pn}_$lab.txt 2>$O/${m}_${pn}_$lab.err
    echo "$m $pn $lab rc=$? $(md5sum < $O/${m}_${pn}_$lab.txt | cut -c1-8)"; done; done
  timeout 600 dotnet $CLI $M8 --provider cuda --logits $D/logits_q8_$lab.bin -p "$P1000 $P1000 $P1000" > $O/logits_q8_$lab.txt 2>&1
  P=$W/bench/PplBench/bin/Release/net11.0/PplBench
  n=1; [ $lab = new ] && n=5
  for r in $(seq 1 $n); do
    $P $M8 --provider cuda --text $T --seq 3072 --chunk 1024 --tokens 12288 --dump $D/q8_${lab}_$r.f16 > $O/ppl_q8_${lab}_$r.txt 2>&1
    $P $M27 --provider cuda --text $T --seq 3072 --chunk 512 --tokens 6144 --dump $D/q27_${lab}_$r.f16 > $O/ppl_q27_${lab}_$r.txt 2>&1
    $P $M27 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/q27s_${lab}_$r.f16 > $O/ppl_q27s_${lab}_$r.txt 2>&1
    echo "$lab run $r: q8 $(md5sum < $D/q8_${lab}_$r.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*$' $O/ppl_q8_${lab}_$r.txt | tail -1); q27 $(md5sum < $D/q27_${lab}_$r.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*$' $O/ppl_q27_${lab}_$r.txt | tail -1); q27 2048/512 $(md5sum < $D/q27s_${lab}_$r.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*$' $O/ppl_q27s_${lab}_$r.txt | tail -1)"
  done
done
echo "logits q8: $(cmp -s $D/logits_q8_base.bin $D/logits_q8_new.bin && echo bit-identical || echo DIFFER)"
for m in q8 q27 q27s; do for r in 1 2 3 4 5; do echo "ppl $m new $r vs base: $(cmp -s $D/${m}_base_1.f16 $D/${m}_new_$r.f16 && echo logits-identical || echo LOGITS-DIFFER)"; done; done
cd $O; for f in *_new.txt; do b=${f%_new.txt}; [ -e ${b}_base.txt ] && { cmp -s $f ${b}_base.txt && echo "same $b" || echo "DIFF $b"; }; done
rm -rf $D
