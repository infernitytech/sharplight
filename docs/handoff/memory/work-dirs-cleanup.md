---
name: work-dirs-cleanup
description: "User rule (2026-11-15): builds/logs/temp in a work directory next to the repos, never /tmp; worktrees on fast local disk; periodically clean up temp and finished worktrees; use the hardware fully"
metadata:
  type: feedback
---

User (2026-11-15): "move your builds etc from tmp to a sibling directory and periodically cleanup also for worktrees so
you don't run out of RAM. Remember this rule." Also: "you have now full reign over the machine and 3 gpus .. run multiple
tests in parallel ... make use of this expensive hardware."

Rules (paths are configurable through tools/agent/env.sh; defaults in brackets):
- Builds, test logs and temp files go to the work directory `$SL_WORK` [`$SL_DATA/work`] (build/, logs/, tmp/), never
  /tmp or a session scratchpad. `TMPDIR` points at local disk (`$SL_LOCAL/work-tmp`).
- After each merge remove the agent's worktree (`git worktree remove`, delete the merged branch) so bin/obj trees don't
  pile up; run the cleanup script periodically (prune worktrees, old temp files; no `du` over a network FS).
- Engine working clone on fast local disk: `$SL_REPO` [`$SL_LOCAL/sl-repo`], remotes `origin` (GitHub) and, where the
  machine has a slower persistent volume, `backup` (bare or `receive.denyCurrentBranch=updateInstead`). Agent worktrees:
  `git -C $SL_REPO worktree add $SL_WORKTREES/<name> -b <branch> develop`, commit often, push the branch to `backup` at
  hand-back. Merge verification runs in a separate detached worktree `$SL_VERIFY` (`git checkout --detach develop`) under
  the verify lock.
- Models: one master store (`$SL_MODELS`), a local-disk mirror for tests (tiny-* copied, the rest symlinked), and big
  checkpoints fetched straight from the Hub to local disk when a network volume is slow. Python references in uv venvs
  on local disk (`$SL_LOCAL/venvs`), never system Python.
- Watch local disk: stay under 90 %, delete downloads > 2 GB after use. Safe to delete: merged worktrees, venvs of
  finished agents, big GGUFs only used by skippable real-model tests.

History (RunPod pod, 2026-11-15..26): repos, tools, models and logs lived on the persistent network volume; clone,
worktrees, verify worktree, SDK copy, venvs and model mirror on local disk. The network FS was slow (down to ~14 MB/min)
and its quota filled twice (2026-11-24 with ~150 GB of stale tool dirs; 2026-11-26), and local disk hit 97-99 % with ~6
agents downloading checkpoints.

**Why:** an expensive machine; avoid filling RAM/disk with stale build trees and keep I/O on fast disks.
**How to apply:** put these paths and the cleanup duty into every agent brief. Related: [[machine-runpod-3gpu]],
[[git-worktrees-parallel]], [[keep-machine-responsive]].
