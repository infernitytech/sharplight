#!/bin/bash
# builds kq harness binaries base/new for formats x shapes x tile rows into $WORK/kq
. <env.sh>
mkdir -p $WORK/kq; cd $TREE/bench/s77_kq
st() { # stages for fmt bb, K, BM, layout(old/new)
  local bb=$1 k=$2 bm=$3 lay=$4
  local al=0; [ $((bb % 16)) = 0 ] && [ $((k / 256 * bb % 16)) = 0 ] && al=1
  local ch; [ $al = 1 ] && ch=$((bb / 16)) || ch=$(((bb + 29) / 16))
  [ $lay = new ] && ch=$(((ch + 1) | 1))
  local s=$((bm * 512 + 64 * ch * 16)); local n=$((98304 / s)); [ $n -gt 4 ] && n=4; echo $n; }
for f in 4 5 6; do case $f in 4) bb=144;; 5) bb=176;; 6) bb=210;; esac
for shape in 4096x12288 12288x4096; do K=${shape%x*}; N=${shape#*x}
for bm in 16 32 64; do for lay in base new; do
  KF=$TREE/src/Sharplight.Shaders/Kernels/gemm_kq_native.cu; [ $lay = base ] && KF=$WORK/kq_base.cu
  s=$(st $bb $K $bm $([ $lay = base ] && echo old || echo new)); wm=1
  nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 -DK=$K -DN=$N -DKQ_FMT=$f -DKQ_BM=$bm -DKQ_STAGES=$s -DKQ_SPLIT=1 -DKQ_WM=$wm -DKFILE="\"$KF\"" bench_kq.cu -o $WORK/kq/kq_${f}_${K}_${N}_${bm}_$lay 2>&1 | grep -i error &
done; done; wait; done; done
ls $WORK/kq | wc -l
