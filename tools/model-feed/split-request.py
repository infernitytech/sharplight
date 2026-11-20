#!/usr/bin/env python3
"""Splits a model-feed publish request into small and large builds (docs/model-feed.md, "Self-hosted feed runner").

Large entries (download plus package above the threshold, or named in the explicit list) build on the self-hosted feed runner;
small ones on a GitHub-hosted runner. Promotions of packages built by the large job run in that job, after the build.

Usage: split-request.py --catalog catalog/verified-models.json --entries "a b:c" --promote "x y@1.0.0"
                        [--threshold-bytes 10000000000] [--large "id-or-spec ..."] [--self-hosted true|false]
Writes key=value lines (GitHub step outputs): small, large, promote_small, promote_large, skipped, and prints a summary to stderr.
"""
import argparse
import json
import sys


def variant_of(entry, variant_id):
    variants = entry.get("variants") or []
    if variant_id is None:
        return next((v for v in variants if v.get("recommended")), variants[0] if variants else None)
    return next((v for v in variants if v["id"] == variant_id), None)


def total_bytes(entry, variant):
    companions = {c["id"]: c for c in entry.get("companions") or []}
    sets = [variant] + [companions[c] for c in variant.get("companions") or [] if c in companions]
    return sum(f.get("size", 0) for s in sets for f in s.get("files") or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--entries", default="")
    parser.add_argument("--promote", default="")
    parser.add_argument("--threshold-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--large", default="", help="entry, entry:variant or package ids that always build on the self-hosted runner")
    parser.add_argument("--self-hosted", default="false")
    args = parser.parse_args()

    with open(args.catalog, encoding="utf-8") as stream:
        catalog = {entry["id"]: entry for entry in json.load(stream)["entries"]}
    forced = set(args.large.split())
    self_hosted = args.self_hosted.strip().lower() == "true"
    small, large, large_ids, notes = [], [], set(), []
    for spec in args.entries.split():
        entry_id, _, variant_id = spec.partition(":")
        entry = catalog.get(entry_id)
        if entry is None:
            print(f"::error::The catalog has no entry '{entry_id}'.", file=sys.stderr)
            return 1
        variant = variant_of(entry, variant_id or None)
        if variant is None:
            print(f"::error::{entry_id} has no variant '{variant_id or '(recommended)'}'.", file=sys.stderr)
            return 1
        package_id = f"{entry_id}.{variant['id']}".lower()
        size = total_bytes(entry, variant)
        # The build needs the sources plus the converted package (about the same size again).
        is_large = size > args.threshold_bytes or {spec, entry_id, package_id} & forced
        (large if is_large else small).append(spec)
        if is_large:
            large_ids.add(package_id)
        notes.append(f"{spec} -> {package_id}: {size / 1e9:.1f} GB, {'self-hosted' if is_large else 'hosted'}")

    promote_small, promote_large = [], []
    for spec in args.promote.split():
        (promote_large if spec.split("@", 1)[0] in large_ids else promote_small).append(spec)

    skipped = []
    if large and not self_hosted:
        skipped = large + promote_large
        large, promote_large = [], []

    for note in notes:
        print(note, file=sys.stderr)
    if skipped:
        print("::warning::No self-hosted feed runner is enabled (repository variable MODEL_FEED_SELF_HOSTED is not 'true' or no runner "
              f"labelled sharplight-feed is online); skipped: {' '.join(skipped)}. The small entries still publish.", file=sys.stderr)
    print(f"small={' '.join(small)}")
    print(f"large={' '.join(large)}")
    print(f"promote_small={' '.join(promote_small)}")
    print(f"promote_large={' '.join(promote_large)}")
    print(f"skipped={' '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
