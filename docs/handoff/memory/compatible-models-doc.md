---
name: compatible-models-doc
description: "User 2026-11-16: keep a public docs/compatible-models.md listing all models tested and compatible (per type, formats, verification); every bring-up/sweep adds rows"
metadata:
  type: project
---

User (2026-11-16): "keep a document with all models tested and compatible... we want that later for people to look at."

- File: docs/compatible-models.md (public, user-facing). Per model type a table: model, Hub repo, formats/quants
  tested, providers (CPU/Vulkan), verification (vs which reference, numbers), notes, date.
- Every bring-up agent and every popularity sweep adds its verified models; put this into every brief; check it at merge.

**Why:** users need to know what runs before downloading.
**How to apply:** only list verified models; mark partial support; keep it current with each merge. Related:
[[core-task-utilities]].

Update 2026-11-16 (user): ship a verified-model catalog with Studio (Discover: "Verified" vs "Hub search", per task, exact
versions/quants, one-click install incl. companions). Single source of truth: catalog/verified-models.json (pinned Hub
revision SHA, exact files, companions, verification); docs/compatible-models.md tables are generated from it. Every
bring-up/sweep must add catalog entries; put this into every brief.

Queued (user 2026-11-17), after the hot-model sweeps: rework the verified catalog / Studio Discover to list mostly official model releases: per entry/variant a source field (official / trusted provider / community); official or trusted variants replace community ones as recommended where verified; community-only entries hidden by default in Discover ("Show community models" filter); Hub search ranks official repos and trusted providers first; Studio Discover highlights (featured models) also switch to official releases. The catalog feeds Studio Discover "Verified models", the website and the server feed alike.
