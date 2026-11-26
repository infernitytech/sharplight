---
name: branching-and-studio-releases
description: "User 2026-11-15: main = stable, develop = unstable integration, release/* branches via Nerdbank.GitVersioning; Studio app packages for Linux/Windows/macOS with Velopack install/update from a Cloudflare R2 bucket"
metadata:
  node_type: memory
  type: project
  originSessionId: 4b250610-6af0-429c-aab7-825cb26032c5
  modified: 2026-11-15T05:14:52.601Z
---

Branching (user, 2026-11-15):
- `main` is the stable branch. `develop` (created 2026-11-15 from main) is the unstable integration branch: every
  worktree agent's branch merges into develop; develop merges into main only when everything is green (build + all suites).
- Releases come from `release/vX.Y` branches (Nerdbank.GitVersioning `nbgv prepare-release`; version.json already has
  `release.branchName = release/v{version}`).

Studio distribution: publish Sharplight Studio for Linux, Windows and macOS as proper app packages, with Velopack
install + auto-update. The user provides a Cloudflare R2 bucket for release and delta/patch files (credentials come
from the user; keep them in GitHub secrets / a local env file outside the repo, never in git).

Decisions 2026-11-15: push `develop` whenever it is green (privacy scan first); main / release/* / tags still only with
the user's go-ahead. Velopack channels: stable (from release/*) and unstable (from develop) for each of win-x64,
win-arm64, linux-x64, linux-arm64, osx-arm64 = 10 channels (e.g. `linux-x64-stable`). Updates are served from a public
custom domain on the R2 bucket (the user tells the domain; keep it configurable); CI uploads with GitHub secrets
R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET. Set on GitHub 2026-11-15: variables STUDIO_UPDATE_URL=https://download.sharplight.dev/studio, STUDIO_UPDATE_PREFIX=studio;
environment `studio-release` (branches develop, release/*) holds the R2 secrets (user adds them). Studio UI: update checks, update-available dialog
with release notes, download progress, restart-to-apply, settings (channel, auto-check, check now).

**Why:** a real release process for the app, not ad-hoc tags from main.
**How to apply:** merge agents into develop, not main; push rules unchanged (only with the user's go-ahead, except
macos/round-1 CI pushes, see [[releases-and-macos]]). Related: [[public-repo-privacy]], [[git-worktrees-parallel]].
