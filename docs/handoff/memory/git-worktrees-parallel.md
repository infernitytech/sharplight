---
name: git-worktrees-parallel
description: "Sharplight is a git repo; parallel work (subagents) goes into git worktrees on their own branches, merged into develop"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-11-13T00:54:04.730Z
---

Sharplight is a git repository (branch main, public at github.com/infernitytech/sharplight since 2026-11-13; history was rewritten for publishing, old history kept locally as legacy-master). Parallel workstreams — e.g. vision support, model bring-up, kernel optimization — run in separate git worktrees on their own branches and are merged back into main (branches based on legacy-master must be cherry-picked onto main); never use ad-hoc copies of the source tree.

**Why:** User asked on 2026-11-13: "please make this a git repository and use worktrees" when parallel agents were started.

**How to apply:** Spawn agents with worktree isolation, have them commit on their branch (no push, no main edits), keep shared-file edits additive; merge and run the full test suites on main afterwards. Commit only what was asked/agreed. Related: [[format-agnostic-core]], [[benchmarking-shared-gpu]].

Update 2026-11-15: the integration branch is `develop` (see [[branching-and-studio-releases]]); agent branches merge into develop, develop into main only when green.
