#!/bin/bash
# usage: prof.sh <out> <model> <rows> -- one profiled prefill (caller holds the lock)
export TMPDIR=${TMPDIR:-/tmp}
export SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
SHARPLIGHT_VK_PROFILE=2 SHARPLIGHT_PROFILE_TOP=600 timeout 900 ${SL_WORKTREE:-.}/samples/Sharplight.Cli/bin/Release/net11.0/sharplight $2 --bench $3 --ctx $(($3+256)) --provider ${PROV:-vulkan} > $1 2>&1
