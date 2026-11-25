---
name: benchmarking-shared-gpu
description: "The workstation GPU is shared with the user's other apps; benchmark with GPU timestamps / min times, not wall-clock means"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-12T22:52:23.479Z
---

The RTX PRO 6000 is shared with the user's own running apps (a "restora" Python process, the "Sloport" dotnet app in ~/Documents/work/fakedreality which periodically uses ~75% SM), so wall-clock decode numbers swing 2-3x between runs.

**Why:** Observed 2026-11-12: identical builds measured 14 vs 29 tok/s depending on Sloport activity.

**How to apply:** Use `sharplight --bench N` best/median plus its "GPU ms/step" (per-step GPU timestamps), per-kernel min times from SHARPLIGHT_VK_PROFILE=1, and dispatch/barrier counts (SHARPLIGHT_DUMP_SCHEDULE=1). Check `nvidia-smi pmon -c 1` before trusting a run. Don't kill the user's processes. Related: [[device-features-fallbacks]].

**Serving numbers need correct text:** `bench/s16_serve.sh` runs `bench/textcheck.py` after its rounds by default (batched greedy
texts of natural prompts at 16 and the slot count against single-request texts; exit code 3 and `*.textcheck.FAIL` on garbage).
Speed rounds alone never read their text: batched Qwen3-8B-FP8 served garbage for twelve rounds unnoticed (benchmarks.md 101.4,
102). Do not report a serving row whose text check failed.
