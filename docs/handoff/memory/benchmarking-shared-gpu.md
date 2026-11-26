---
name: benchmarking-shared-gpu
description: "The workstation GPU is shared with the user's other apps; benchmark with GPU timestamps / min times, not wall-clock means"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-12T22:52:23.479Z
---

Previous setup (until 2026-11-15): the owner's workstation RTX PRO 6000 was shared with the owner's own running applications (one of them periodically used ~75% SM), so wall-clock decode numbers swung 2-3x between runs. The rule applies to any GPU that other processes (other agents, the owner's apps) may use.

**Why:** Observed 2026-11-12: identical builds measured 14 vs 29 tok/s depending on the other application's activity.

**How to apply:** Use `sharplight --bench N` best/median plus its "GPU ms/step" (per-step GPU timestamps), per-kernel min times from SHARPLIGHT_VK_PROFILE=1, and dispatch/barrier counts (SHARPLIGHT_DUMP_SCHEDULE=1). Check `nvidia-smi pmon -c 1` before trusting a run. Don't kill processes you did not start. Related: [[device-features-fallbacks]].

**MoE decode comparisons must be like-for-like (found 2026-11-21, benchmarks §64):** BatchBench with fixed random token ids routes MoE far more narrowly than generated text. On gpt-oss-20b this overstated the decode step by ~20% (we reported "1.00x of vLLM" when it was really 0.81x). Always use BatchBench `--feedback` or `--text` against `s54_vllm_decode_profile.py --words` (a CUDA venv with vLLM), and add our LM head to vLLM's step range.
