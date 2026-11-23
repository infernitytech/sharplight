#!/bin/bash
# correctness (greedy text, PplBench dumps) base vs branch, then Cuda.Tests and Serving.Tests
. <out>/env.sh
O=$OUT/correct; mkdir -p $O $OUT/tests; D=$WORK/ppl; mkdir -p $D
export XDG_CACHE_HOME=$WORK/cache_c
P40="Write a detailed essay about the role of water and power in modern history. Explain the history, give concrete examples and discuss open questions."
P1000="$(cat $OUT/p1000.txt)"; T=$WORK/wikitext2_test.txt
for lab in base new; do [ $lab = base ] && W=$BASE || W=$TREE
  CLI=$W/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll
  for m in Qwen3-8B-FP8 gpt-oss-20b; do for pn in p40 p1000; do [ $pn = p40 ] && P="$P40" || P="$P1000"
    timeout 900 dotnet $CLI ~/models/hf/$m --provider cuda -n 120 -p "$P" > $O/${m}_${pn}_$lab.txt 2>$O/${m}_${pn}_$lab.err
    echo "$m $pn $lab rc=$? $(md5sum < $O/${m}_${pn}_$lab.txt | cut -c1-8)"; done; done
  PB=$W/bench/PplBench/bin/Release/net11.0/PplBench
  n=1; [ $lab = new ] && n=2
  for r in $(seq 1 $n); do
    $PB $M8 --provider cuda --text $T --seq 3072 --chunk 1024 --tokens 12288 --dump $D/q8_${lab}_$r.f16 > $O/ppl_q8_${lab}_$r.txt 2>&1
    echo "ppl q8 $lab $r $(md5sum < $D/q8_${lab}_$r.f16 | cut -c1-8) $(grep -o 'PPL [0-9.]*' $O/ppl_q8_${lab}_$r.txt | tail -1)"
  done
done
cd $TREE; TEST_HOME=<test-home>
HOME=$TEST_HOME nice -n 15 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build > $OUT/tests/cuda.txt 2>&1; echo "Cuda.Tests: $(grep -o 'Total: .*' $OUT/tests/cuda.txt | tail -1)"
HOME=$TEST_HOME nice -n 15 dotnet run --project tests/Sharplight.Serving.Tests -c Release --no-build > $OUT/tests/serving.txt 2>&1; echo "Serving.Tests: $(grep -o 'Total: .*' $OUT/tests/serving.txt | tail -1)"
rm -rf $D
