#!/bin/bash
# usage: mkv.sh <name> <kernel.cu> <cfg: gu|dn|g> [data: r|v] [extra -D...]
# Builds b_<name>: the harness (bench_mc.cu) launching <name>.cubin from the engine's NVRTC 13.0 (copy warpgroup, uniform weight scales).
#   gu: gate/up dual 4096 -> 2 x 12288 with the E4M3 store (QOUT, the model's form); dn: down 12288 -> 4096; g: 4096 -> 24576 f32 (CUTLASS's shape)
#   data r: random codes; v: vLLM's operands (LOADTAG $VD/<gu|dn>)
cd <harness-dir>
n=$1; kf=$(readlink -f $2); cfg=$3; data=${4:-r}; shift 4
case $cfg in
  gu) D="-DK=4096 -DN=12288 -DDUAL=1 -DQOUT=1 -DQCAP=8192"; tag=gu ;;
  dn) D="-DK=12288 -DN=4096 -DDUAL=0"; tag=dn ;;
  g)  D="-DK=4096 -DN=24576 -DDUAL=0"; tag=gu ;;
esac
D="$D -DF8N_WG=1 -DF8N_WUNI=1 $*"
L=""; [ $data = v ] && L="-DLOADTAG=\"${VD:-<vd>}/$tag\""
cat > w_$n.cu <<EOW
#define XCAP 8192
#define Y_STRIDE N
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a) * 1e-4f)) * (b))
#define SL_FAMILY_ARCH 1
#include "$kf"
EOW
LD_LIBRARY_PATH=<nvrtc-lib-dir> nice -n 15 python3 -I nvrtc.py w_$n.cu $n.cubin $D || exit 1
cuobjdump -sass $n.cubin > $n.sass
nice -n 15 nvcc -O3 -gencode arch=compute_120f,code=sm_120f -std=c++17 -diag-suppress 177,550,1650 -w $D -DKFILE="\"$kf\"" $L \
  -DMS_A=2048 -DMS_B=1920 -DCHECK_T=2048 -DSUSTAIN=${SUSTAIN:-1500} -DTMA -DPERSIST=82 -DKENTRY=gemm_fp8_tma -DWUNI \
  -DPTXFILE="\"<harness-dir>/$n.cubin\"" -DPTXENTRY="\"gemm_fp8_tma\"" -I<tree>/bench/s86_bs bench_mc.cu -o b_$n -lcuda -lnvidia-ml || exit 1
echo "built b_$n: QMMA $(grep -c QMMA $n.sass), .reuse $(grep QMMA $n.sass | grep -c reuse), FFMA $(grep -c FFMA $n.sass), local $(grep -cE 'LDL|STL' $n.sass)"
