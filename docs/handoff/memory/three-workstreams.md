---
name: three-workstreams
description: "User 2026-11-18: keep constant progress on all three streams (engine, Studio, Server) in parallel; engine gets the GPUs, server mostly build/CPU; overlap long test runs; watch budget"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9328121f-dff2-4ed6-aa29-9b5940aae31c
  modified: 2026-11-18T03:49:41.313Z
---

User (2026-11-18): "Try to assume all work streams studio/server/engine. The engine gets most gpu resources for bringup
and performance work. The server is mostly software engineering. So it should mostly use build resources... We want
constant progress on all fronts... through overlapping work ... long test runs that one stream has overlap with other
work ... so we get done faster." Also: "stay careful with the budget".

**Why:** the previous session let the Server (private repo infernitytech/sharplight-server, see [[sharplight-server-product]])
sit idle while only engine agents ran.
**How to apply:** always keep at least one agent per stream: engine (bring-up + perf, owns the GPUs; perf agent gets a
GPU exclusively), Studio (UI work, light GPU), Server (CPU/build only, no GPU unless a real-model test needs it briefly).
Stay within [[keep-machine-responsive]] (~6 agents, ≤3 builds) — prefer cheaper models (sonnet) for server/studio
engineering, opus for hard parity/perf debugging. When a slot frees, refill it with the stream that has nothing running.

User 2026-11-19 asked "what are you doing on the vllm performance gap?" after a day of only bring-ups/fixes on a single-GPU machine: keep one perf agent running continuously (it gets GPU priority for its final benchmark runs); don't let perf lapse because the GPU is scarce.
