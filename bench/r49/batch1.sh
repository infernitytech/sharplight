#!/bin/bash
S="4096,12288;12288,4096"
uptime
echo "### shipped"; ./mmraw.sh f8e4m3_b128 2048 "$S"
echo "### nv2 mapelement"; SL_PROBE_NV2=1 ./mmraw.sh f8e4m3_b128 2048 "$S"
echo "### nv2 bcast"; SL_PROBE_NV2=1 SHARPLIGHT_GEMM_MACROS=NV8_BCAST=1 ./mmraw.sh f8e4m3_b128 2048 "$S"
echo "### nv2 noscale"; SL_PROBE_NV2=1 SHARPLIGHT_GEMM_MACROS=NV8_NOSCALE=1 ./mmraw.sh f8e4m3_b128 2048 "$S"
