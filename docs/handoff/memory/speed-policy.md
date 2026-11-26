---
name: speed-policy
description: "IMPORTANT (user, repeated 2026-11-16): time is of the essence; land features/fixes fast: merge each green branch immediately, verify with build + style + affected suites only, trust agents' full runs; plus 2026-11-15: speed up work: no expensive full test runs per change; targeted tests while iterating, full suites once per agent hand-back and once per merge batch; lean builds"
metadata:
  type: feedback
---

User (2026-11-15): "try to speed up work in general, maybe don't run expensive test runs for each change, and rather in
bulk for changes".

- Agents: build only touched projects with `-m:6` (native builds `-j 6`), run only affected test classes while
  iterating, full suites once before hand-back; commit in logical chunks; delete big temp outputs promptly.
- Merges: batch several finished branches, then one full local suite run per batch; fast suites every time, slow ones
  (ONNX Conformance ~15 min, Families ~10 min, full Vulkan) once per batch. GitHub CI on develop runs the CPU suites
  for free, so locally the GPU suites matter most.
- Cleanup: keep periodic cleanup fast (no `du` over a network FS; the user interrupted a slow one).

**Why:** the container has only ~30 CPU cores for ~13 agents; compile and test time dominate.
**How to apply:** include the build/test policy in every agent brief. Related: [[work-dirs-cleanup]], [[machine-runpod-3gpu]] (previous machine).
- User 2026-11-15: do NOT push agent branches to GitHub for CI (only develop / macos/round-1 as already allowed).
- User 2026-11-15: local model mirror approved: tools/agent/sync-local-models.sh copies tiny-* models from the master
  model store to a local-disk mirror (rest symlinked); tests run with HOME pointed at a test home whose `models` links to
  the mirror, via tools/agent/sl-test <Project> or tools/agent/run-suites.sh. Re-sync after model changes.
- User 2026-11-16 ("time is of the essence, land features and fixes faster"): merge verification = build + style checks
  + only the affected suites (minutes); trust the agents' own full-suite runs; merge and push each green branch right
  away instead of batching; full local suite runs only for risky runtime/kernel changes or occasionally.

**Repeated emphatically by the user 2026-11-16: "remember your new faster work method, and keep it up, this is important."**
Workflow: agent reports green → privacy scan → merge into develop → build + style checks + affected suites (minutes) →
push immediately → refill the slot. No full-suite reruns per merge; no waiting to batch.

Merge check also validates .github/workflows/*.yml with yaml.safe_load (2026-11-16: an unquoted "name: video: ..." in native.yml silently broke Studio/CI/Packages for 3 pushes) and glances at `gh run list --branch develop` after each push.

Lesson 2026-11-17 (verify worktree lock): background merge/check chains must (1) never share the verify worktree concurrently — take `flock $SL_STATE/.sl-verify.lock` around checkout+build+tests; (2) wait on a marker unique to the current run (not a grep for "ok=1" that may match an earlier run printed into the same log); (3) never edit a script while it runs (bash reads incrementally). A chain once pushed Metal r4 after only Core/Cpu because of (1)+(2).

Lesson 2026-11-18: an MSBuild node-reuse server started inside `flock` inherits the lock fd and holds the verify lock forever (verify chain blocked 30 min). In verify scripts build with `-nodeReuse:false` and `9>&-`. Test project names in run-suites.sh need the `.Tests` suffix (e.g. `Vulkan.Tests`), else "file path does not exist".
