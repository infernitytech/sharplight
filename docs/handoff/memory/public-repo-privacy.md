---
name: public-repo-privacy
description: Sharplight is public on GitHub; commits must use the noreply email and no /home/<user> paths or personal emails may be committed
metadata:
  type: project
---

Sharplight is published publicly at github.com/infernitytech/sharplight (MIT, default branch main, 2026-11-13). The history was rewritten to the GitHub noreply address 826745+BernhardGlueck@users.noreply.github.com and home paths were scrubbed; the repo-local git config now sets that email.

**Why:** The user chose to rewrite history for privacy before publishing. Commits made with the global personal identity, or tracked build artifacts like .pyc files with absolute paths, would leak personal data again.

**How to apply:** Before every push, check that new commits use the noreply author and committer (`git log origin/main..main --format='%ae %ce'`) and grep the diff for home paths and personal emails. Push only when the user asks. Related: [[git-worktrees-parallel]].
