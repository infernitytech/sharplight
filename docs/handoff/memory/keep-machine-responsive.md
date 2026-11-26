---
name: keep-machine-responsive
description: "HARD RULE (user 2026-11-15): never load the machine so much that the user can't give input to Claude; agent cap sized to the machine (was ~6, later ~3), builds/tests niced, -m:4"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 4b250610-6af0-429c-aab7-825cb26032c5
  modified: 2026-11-15T09:39:27.726Z
---

User (2026-11-15), after the machine got so loaded that their input didn't get through: "be careful not to block the
whole machine so that I cannot even give input to you (that happened before). Remember this." and "Please be careful
not to totally overwhelm the machine."

- At most ~6 concurrent agents on the 30-core pod (~3 on the 10-core one); at most ~3 of them building/testing at the same time. Refill freed slots, never exceed. On a new machine derive the caps from its CPU quota, RAM limit and GPU count.
- Every build/test command runs niced: `nice -n 15 ionice -c3 dotnet build ... -m:4` (native builds `-j 4`), the same
  for `dotnet run` test suites and Python reference runs. My own merge checks too (run-suites.sh does it).
- Prefer targeted tests; full suites once per hand-back / merge batch ([[speed-policy]]).
- Check `uptime` / cgroup throttling before starting more work; if load is far above the CPU quota (cgroup cpu.max, not nproc), pause or don't refill.

**Why:** the container has a ~30-core CPU quota; 14+ agents compiling starved the interactive session.
**How to apply:** put the nice/-m:4 rule and the agent cap into every agent brief and resume message. Related:
[[machine-runpod-3gpu]] (previous machine), [[work-dirs-cleanup]].

Agents must never kill processes by broad patterns (`pkill -f "dotnet build ..."` killed other agents' builds on 2026-11-16); only PIDs they started. Put this into every brief. Also: never set user.email/GIT_AUTHOR_EMAIL in agents (a speech agent committed as noreply@anthropic.com; fixed by squash before push) — check authors of every branch before merging.

Enforced since 2026-11-17: tools/agent/env.sh defines a `dotnet` shell function that makes `dotnet build|pack|publish` wait for one of `SL_BUILD_SLOTS` flock slots (`$SL_STATE/.sl-build-slots`); agents get it by sourcing env.sh. Load had reached 40-60 with ~8 parallel builds. Size the slot count and the agent cap to the machine (on the 10-core pod: 2 slots, ~3 agents; on the 30-core pod: 3 slots, ~6 agents).

Lesson 2026-11-18: an agent's Glob for `*.slm` over `/` ran 4 h (bfs PID in D state) and saturated the network file system holding the repos and models: every process touching it hung (even `echo` in agent shells), one agent produced nothing. Agent rules (docs/handoff/README.md, "Agent rules") forbid recursive searches over `/`, the home directory and the persistent data volume. Killing it needed the user (auto-mode classifier blocked my kill).
