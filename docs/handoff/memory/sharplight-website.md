---
name: sharplight-website
description: "Public website repo infernitytech/sharplight-website : Jekyll on GitHub Pages; promotes OSS engine + Studio downloads, docs, calm Server marketing"
metadata:
  type: project
---

User (2026-11-17): build the Sharplight website as a Jekyll site in the public repo infernitytech/sharplight-website
(cloned next to the engine repo), published via GitHub Pages; the user allowed pushing it.
Content: promote the open-source engine (NuGet docs for developers) and Sharplight Studio (downloads with automatic
platform detection, overridable), documentation, and calm marketing for the enterprise Sharplight Server (benchmark
graphs, cluster capabilities, broad model compatibility). Styling follows Sharplight Studio. Sensible, non-boastful
open-core marketing based on research of how other OSS products with paid add-ons do it.
Strategy/copy via Fable; implementation via cheaper models.

Imprint: company details provided by the owner on 2026-11-17 and approved for publication on the website (imprint.md in
the website repo is the source; bank details deliberately not published). Server early-access contact: the company
contact address on the imprint page.

Offerings (user 2026-11-17): besides Sharplight Server, enterprise inference hardware bundled with the software, and consulting (AI use cases, deployment, integration); present calmly on the enterprise pages, no invented prices/partners/customers.

Decisions (2026-11-17): domain sharplight.dev; open-source page states current MIT license only (no forward-looking stewardship promise); Server pricing contact-only; hardware + consulting available now and independent → /enterprise/ page.

Copy feedback (user 2026-11-17): concise like popular OSS sites (no explaining what developers know, state facts once); one unified /enterprise/ page for deciders with headlines ease of use, security/data stays in-house (DSGVO/GDPR), cost; tech details secondary.

Domain: the user connects sharplight.dev only when the website is completely finished; until then url/canonical/sitemap use the github.io Pages URL, switch is one config change at go-live.

Confirmed claims (user 2026-11-17): "installed in minutes" and "no per-token fees" may be used; software/hardware/consulting each available on its own.

Benchmarks (user 2026-11-17): apples to apples only on the website: same model files/precision, same procedure and statistic for every engine; no notes about a shared machine.

**Why:** public face of the technology and the commercial product.
**How to apply:** never publish private Server internals (only customer-facing capabilities, "early access");
benchmarks with methodology and caveats; privacy scan before every push. Related: [[sharplight-server-product]].
