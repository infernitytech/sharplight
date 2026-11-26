---
name: feed-runner
description: "Self-hosted GitHub Actions runner (org infernitytech, label sharplight-feed) for large model-feed packages; was on the RunPod pod, must be re-registered on a new machine (owner provides the token)"
metadata:
  type: project
---

The owner chose a self-hosted runner for building and publishing large model-feed packages (2026-11-20). The packages are
18-111 GB (some catalogued models are 250-566 GB), and ubuntu-latest has only about 14 GB of free disk.

**Registration rules:**
- Registered at the ORG level (https://github.com/infernitytech), because the owner's token was an org token. The org's
  runner group must allow the public engine repo.
- Labels self-hosted / Linux / X64 / sharplight-feed. The model-feed workflow sends large entries to this label only for
  pushes or dispatches on develop and only when the repository variable MODEL_FEED_SELF_HOSTED is 'true';
  MODEL_FEED_WORK_DIR points at a large disk on the runner.
- Run it as a dedicated non-root user with its work dir on a big disk; aws-cli installed; dotnet comes from setup-dotnet in
  the job. Without systemd (containers) start it with `nohup ./run.sh`; the bootstrap script restarts it when configured.
- Never point pull_request workflows at this runner (public repo).
- Registration tokens come from the owner. Never fetch or create tokens yourself.

**History:** runner `runpod-feed-1` ran on the RunPod pod 2026-11-20..26. With the 2026-11-26 move it is gone: a new
machine needs a new registration (owner provides the org runner token; `config.sh --url https://github.com/infernitytech
--labels sharplight-feed --work <big disk>/_work`; docs/model-feed.md "Self-hosted feed runner").

**Why:** the owner's explicit choice, picked from the self-hosted / larger runners / upstream-reference options.
Related: [[bringup-catalog-feed]], [[releases-and-macos]].
