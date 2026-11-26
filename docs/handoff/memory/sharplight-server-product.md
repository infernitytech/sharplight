---
name: sharplight-server-product
description: "Commercial closed-source Sharplight Server (enterprise on-prem inference platform) in private repo infernitytech/sharplight-server; consumes Sharplight via NuGet"
metadata:
  type: project
---

User (2026-11-17): new commercial product Sharplight Server. Private GitHub repo infernitytech/sharplight-server (cloned next to the public engine repo). Same .NET/C# conventions (docs/code-style.md),
consumes Sharplight only via NuGet packages; Sharplight stays independent (generic functionality is contributed upstream).
Remove the public samples/Sharplight.Server sample (after harvesting reusable parts into the framework).

MVP: Blazor Server, modular monolith (modules register via DI, contribute APIs/UI/EF model), Studio styling shared with
the web UI (change both), clustering with capacity-based placement + routing + model distribution between nodes,
resilience, setup wizard (+ export for scripted node joins), model feed on R2 (.slm + manifest, semver, deps, min server
version, country of origin...; "NuGet for models"), enterprise auth (AD/LDAP, OIDC/SAML, password), roles (Cluster/
Account/Space Admin, Space User; extensible), PostgreSQL for everything (embedded PG option, pgvector + PostGIS, queues),
domain Account > Space > endpoints (OpenAI/Anthropic), chat UI like Claude web, tools/MCP, API keys, per-user privacy.
Phase 2: RAG with ACL-aware connectors. Phase 3: hosted durable agent runtime.

Isolation (user 2026-11-17): OpenAI/Anthropic APIs strictly isolated per account and space: keys bound to one space,
space-local model aliases, prefix/KV cache partitioned per space (no cross-tenant side channel), stored state/usage/
quotas/tools per space, enforced also in the DB (RLS). The user allowed pushing the private repo (2026-11-17).

Embedded PostgreSQL is single-node only (user 2026-11-17); an external PostgreSQL server is always selectable; node join refuses embedded-DB clusters (migrate to external first).

Scale requirement (user 2026-11-17): thousands of accounts, tens of thousands of spaces, thousands of nodes — design for it now (docs/scaling.md in the private repo).

Feed signing key (created 2026-11-17 with the user's explicit go): ECDSA P-384, private key only as GitHub secret MODEL_FEED_SIGNING_KEY (env studio-release, repo infernitytech/sharplight); no other copy exists; public key committed in the engine repo as the trusted key (tools/model-feed/keys/sharplight-feed.pub.pem); rotation = add a new trusted key; offline root key later.

Model feed channels (user 2026-11-17): stable and unstable indexes over the same immutable package files (stored once); new versions → unstable, promote → stable; servers default to stable.

**Why:** diversify into a commercial product on top of the engine.
**How to apply:** design keeps phases 2/3 as separate modules; planning with Fable, implementation with cheaper models;
the private repo is closed source: never copy its code into the public repo; privacy scan applies to both. Related:
[[core-task-utilities]], [[compatible-models-doc]].

Server builds (user 2026-11-18): CI uploads server packages to the same Cloudflare R2 bucket as Studio but under its own prefix (`server/`, next to `studio/` and `models/`), per channel (unstable from develop, stable from release/*). "I always want to test the same builds users would get" — test the R2-published artifacts, not local builds.
Server repo setup 2026-11-18 (user decisions): environment `server-release` (develop + release/* only) holds the R2 secrets (set by me from values the user gave; never stored elsewhere); vars SERVER_UPDATE_PREFIX=server, SERVER_DOWNLOAD_URL=https://download.sharplight.dev/server; public download approved; branch model identical to the engine (main stable, develop integration = unstable channel, release/vX.Y stable) with NBGV.
Server packaging (2026-11-18, feature/packaging): minisign-signed manifests (secret MINISIGN_SECRET_KEY in server-release, public key packaging/keys/sharplight-server.pub), multi-arch GHCR image (private), deb/rpm via nfpm, develop builds against engine source, release/* pinned to nuget.org engine releases. User: "far pre release, develop builds only matter for now" — stable-release hardening items live in the server repo's docs/releasing.md "Before the first stable release" checklist; don't work on them until a stable release is planned.
Q10 decided (user 2026-11-18): conversations owner-only; admins see counts/dates only unless the space enables "Administrators may export the content of conversations"; per-space retention; exports audited — "perfect".
M7 decisions (user 2026-11-18): Q3 licensing = signed offline license file with soft enforcement (editions/limits/grace still to define); audit retention per account, configurable; no SIEM export in v1; /metrics default bearer token or same-machine is fine; Q12 confirmed: no telemetry to the vendor in v1 (OTLP only to the customer's own collector, opt-in).
Audit access (user 2026-11-18): cluster admin only by default; account-admin view/export behind a setting, off by default.
User 2026-11-19: server stream next = go over everything: refactor, remove dead code, and test the UI (instead of M8/RAG for now).
User 2026-11-19: licensing editions table approved (Community 1 node/2 GPU/10 users no expiry; Evaluation 60d/14d grace 2/8/25; Standard 365d/30d 4/16/250; Enterprise 365d/60d unlimited; Community without RAG/agents) — keep everything easily changeable (config/editions.json), mark as approved-but-adjustable.
