#!/bin/bash
# Section 118.2 correction (#62): last-token logits of ~800-token prompts against the CPU provider, Vulkan `fe` with sinks (default)
# and `fa` (SHARPLIGHT_ATTN_FE_SINKS=0), one prompt per 3,500-character slice of docs/benchmarks.md (newlines -> spaces, quotes removed).
#   MODEL=<gpt-oss-20b HF dir> bench/s118_logits_sweep.sh [offset ...]
SL=samples/Sharplight.Cli/bin/Release/net11.0/sharplight
MODEL=${MODEL:?set MODEL}
OUT=${OUT:-/tmp/s118_logits}
mkdir -p "$OUT"
S="$(tr '\n' ' ' < docs/benchmarks.md | tr -d "\"'")"
offsets=("$@"); [ ${#offsets[@]} -eq 0 ] && offsets=(0 5000 10000 20000 40000 60000 80000 100000)
for off in "${offsets[@]}"; do
  P="${S:$off:3500} Summary of the above:"
  $SL "$MODEL" --cpu -p "$P" --logits "$OUT/$off.cpu" > /dev/null 2>&1
  SHARPLIGHT_ATTN_FE_SINKS=1 $SL "$MODEL" --provider vulkan -p "$P" --logits "$OUT/$off.fe" > /dev/null 2>&1
  SHARPLIGHT_ATTN_FE_SINKS=0 $SL "$MODEL" --provider vulkan -p "$P" --logits "$OUT/$off.fa" > /dev/null 2>&1
  python3 - "$OUT/$off" "$off" <<'PY'
import sys, numpy as np
b, off = sys.argv[1], sys.argv[2]
c = np.fromfile(b + ".cpu", np.float32)
r = []
for k in ("fe", "fa"):
    d = c - np.fromfile(b + "." + k, np.float32)
    r.append(f"{k} rms {np.sqrt((d * d).mean()):.4f} max {abs(d).max():.3f}")
print(f"bench@{off} " + " | ".join(r))
PY
done
