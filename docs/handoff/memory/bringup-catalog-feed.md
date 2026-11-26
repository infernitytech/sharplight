---
name: bringup-catalog-feed
description: "Bring-ups must verify the OFFICIAL model release (not community conversions) and add it to the verified catalog, the model feed and Studio"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 00ab6798-7129-46be-a328-7d157af0d23e
  modified: 2026-11-25T23:06:59.118Z
---

Owner 2026-11-25 (restating an existing rule, annoyed it was missed): "when you bring up an official model add its checkpoint to the verified models and to our feed. SeedVR2 is nowhere to be found." and "This was already a rule and you forget we want the official model releases working."

Context: SeedVR2 7B parity/timings (§108) were measured on a ComfyUI community fp16 conversion (numz/SeedVR2_comfyUI), not ByteDance's official .pth; only 3B was in the catalog; neither in the feed.

**Why:** testers and customers use the official releases; community conversions can differ (numz 3B fp16 is 0.8-3 % off the official weights). A bring-up that isn't verified on the official files, in the catalog, in the feed and visible in Studio isn't done.

**How to apply:**
- Verification and benchmarks use the official release files (pinned official repo + revision). Community conversions (fp16/fp8/GGUF) are extra variants, verified separately, never a substitute.
- Every bring-up brief and merge check requires: official checkpoint verified vs the official code; catalog entry per official variant (revision, checksums, licence, verified level) + catalog tool validate/docs/schema; feed publish request entry (large ones via [[feed-runner]]); Studio shows it and installs it with one click and opens the right page.
- Owner 2026-11-26: "We want official releases on our model feed only. Maybe with some community quants as options for lower precision." Feed packages must be built from the publisher's own repo (gated ones too: the feed workflow needs an HF token with accepted terms). Ungated mirrors (e.g. an SD3.5 Medium mirror) are never a feed source. Community quants (GGUF etc.) only as extra lower-precision options next to an official variant.
- Owner 2026-11-26 exceptions: GitHub-only publishers (e.g. Silero VAD) are fed from their official GitHub release (generic GitHub source type); releases whose publisher deleted them (SD 1.5 via stable-diffusion-v1-5 preservation copy) may use the byte-identical preservation copy.
- Related: compatible-models-doc (official releases first in catalog/Discover).
