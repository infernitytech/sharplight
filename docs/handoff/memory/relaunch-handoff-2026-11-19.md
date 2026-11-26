---
name: relaunch-handoff-2026-11-19
description: "HISTORY: the 2026-11-19 pod relaunch (bootstrap script, backup remote); superseded by the 2026-11-26 handoff in docs/handoff/README.md"
metadata:
  type: project
---

**Superseded 2026-11-26:** the current state, branch list and work queue are in docs/handoff/README.md. This note is
history.

The pod was relaunched on 2026-11-19 (larger local disk; it came back with 1 GPU and ~10 cores). Everything outside the
persistent volume was wiped. A bootstrap script (now tools/agent/bootstrap-local.sh, path-generic) rebuilt the local
state: SDK, tools, shell env, gh/git settings from a private secrets folder (never in a repo), the local engine clone with
a `backup` remote on the persistent volume, the detached verify worktree, Claude memory from a backup folder, Python venvs
from pinned `pip freeze` snapshots, dev PostgreSQL for the server tests, Playwright browsers, the local tiny-model mirror.
`--prepare-relaunch` pushed every local branch to the backup remote and listed uncommitted work before the wipe.

At that relaunch `cuda/provider-r3` was merged and `cuda/provider-r4` (FP8 grouped scales) was finished afterwards; the
queue then held Stable Audio and Cohere bring-ups, the Studio license-accept dialog, the Sharplight.Cuda -> Runtime NuGet
dependency, server refactoring and flaky-test fixes (all done since, see the README queue).

**Lesson:** before any machine move or wipe, push every branch somewhere persistent (backup remote or origin), commit or
copy dirty worktrees, and back up the memory notes. A docs/handoff/ copy of the memory is the durable version.
