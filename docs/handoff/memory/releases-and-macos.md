---
name: releases-and-macos
description: Release flow (nbgv tags publish NuGet + Studio AppImage) and the macOS work kept on a separate branch until the user decides
metadata:
  node_type: memory
  type: project
  modified: 2026-11-14T17:07:06.553Z
---

Releases (set up 2026-11-14): Nerdbank.GitVersioning (version.json `0.1-alpha`, SemVer 2). A pushed `v*` tag (`nbgv tag`)
runs packages.yml (16 NuGet packages to nuget.org via trusted publishing, repo variable NUGET_USER, environment `nuget`)
and studio.yml (Studio AppImage + tar.gz attached to the GitHub release). First release: v0.1.2-alpha (NuGet only).

The user wants a **Linux-only Studio release** with all fixes plus the Studio features in progress on 2026-11-14 (.slm
inspector in the library, pre-download compatibility badge); tag it once those are merged and the user says push.

**macOS support is kept separate**: agent work on remote branch `macos/round-1` (MoltenVK bundled as
Sharplight.Vulkan.MoltenVK, macOS/ARM64 CI, NEON CPU kernels, Studio .app). Do not merge into main until the user
decides. Native Metal provider is the later performance path; CoreML only optional for ONNX/ANE.

**Why:** user ships Linux first; no Mac available locally, so macOS is tested on GitHub macOS runners.
**How to apply:** never fold macOS changes into main or a release tag without the user's go-ahead; see
[[public-repo-privacy]] and [[autonomous-loop-workstreams]].
