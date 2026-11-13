#!/usr/bin/env python3
"""Builds the ONNX operator coverage matrix (Markdown) from the onnx schema registry and conformance reports.

Inputs:
  * the operator schemas of the installed onnx package (every domain, every since_version),
  * one or more conformance reports written by tests/Sharplight.Onnx.Conformance (SHARPLIGHT_CONFORMANCE_REPORT=<file>,
    JSON lines with test, provider, status, message, skip, ops) and the `<file>.supported` list of operators the importer accepts,
  * optional notes (tools/onnx-coverage-notes.tsv: `<domain.op>\t<note>`), e.g. known limitations.

Usage (in a venv with onnx, never the system Python):
    python -I tools/onnx-coverage.py --report cpu.jsonl --report vulkan.jsonl [--notes tools/onnx-coverage-notes.tsv] > matrix.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import onnx
import onnx.defs

DOMAINS = ["ai.onnx", "ai.onnx.ml", "ai.onnx.preview", "ai.onnx.preview.training"]


def load_reports(paths):
    results = collections.defaultdict(dict)  # test -> provider -> record
    supported = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                results[r["test"]][r["provider"]] = r
        if os.path.exists(path + ".supported"):
            with open(path + ".supported") as f:
                supported |= {l.strip() for l in f if l.strip()}
    return results, supported


def qualified(domain, op):
    return op if domain == "ai.onnx" else f"{domain}.{op}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", default=[], help="conformance report (JSON lines); repeat per provider")
    parser.add_argument("--notes", help="TSV of per-operator notes")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    schemas = collections.defaultdict(list)
    for s in onnx.defs.get_all_schemas_with_history():
        schemas[(s.domain or "ai.onnx", s.name)].append(s)
    results, supported = load_reports(args.report)
    providers = sorted({p for per in results.values() for p in per})
    notes = {}
    if args.notes and os.path.exists(args.notes):
        with open(args.notes) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, note = line.rstrip("\n").split("\t", 1)
                    notes[key] = note

    # Tests per operator: a test counts for every op its model uses (node tests usually have one op, expanded tests a few).
    tests_of = collections.defaultdict(set)
    for test, per in results.items():
        for record in per.values():
            for op in record.get("ops", []):
                tests_of[op].add(test)

    print(f"Operator schemas of onnx {onnx.__version__} (ai.onnx opset {onnx.defs.onnx_opset_version()}).\n")
    print("| domain | operators | importer | " + " | ".join(f"{p} all tests pass" for p in providers) + " |")
    print("|---|---|---|" + "---|" * len(providers))
    for domain in DOMAINS:
        ops = sorted(name for (d, name) in schemas if d == domain)
        imported = sum(1 for op in ops if qualified(domain, op) in supported)
        cells = []
        for p in providers:
            with_tests = [op for op in ops if tests_of.get(qualified(domain, op))]
            passing = [op for op in with_tests if all(results[t].get(p, {}).get("status") == "Pass" for t in tests_of[qualified(domain, op)] if p in results[t])]
            cells.append(f"{len(passing)} / {len(with_tests)} with tests")
        print(f"| {domain} | {len(ops)} | {imported} | " + " | ".join(cells) + " |")
    print()
    print("| provider | node tests | pass | wrong | import fail | compile fail | run fail | harness unsupported |")
    print("|---|---|---|---|---|---|---|---|")
    for p in providers:
        counts = collections.Counter(per[p]["status"] for per in results.values() if p in per)
        total = sum(counts.values())
        print(f"| {p} | {total} | {counts['Pass']} ({100 * counts['Pass'] / max(total, 1):.1f}%) | {counts['Wrong']} | {counts['ImportFail']} | "
              f"{counts['CompileFail']} | {counts['RunFail']} | {counts['HarnessUnsupported']} |")
    if args.summary_only:
        return

    for domain in DOMAINS:
        print(f"\n### {domain}\n")
        print("| operator | opset versions | importer | " + " | ".join(f"{p} tests" for p in providers) + " | notes |")
        print("|---|---|---|" + "---|" * len(providers) + "---|")
        for (d, name) in sorted(k for k in schemas if k[0] == domain):
            key = qualified(domain, name)
            versions = sorted(s.since_version for s in schemas[(d, name)])
            deprecated = any(s.deprecated for s in schemas[(d, name)] if s.since_version == versions[-1])
            cells = []
            for p in providers:
                tests = [t for t in tests_of.get(key, ()) if p in results[t]]
                passed = sum(1 for t in tests if results[t][p]["status"] == "Pass")
                cells.append(f"{passed}/{len(tests)}" if tests else "-")
            imported = "yes" if key in supported else ("deprecated" if deprecated else "no")
            print(f"| {name} | {', '.join(map(str, versions))} | {imported} | " + " | ".join(cells) + f" | {notes.get(key, '')} |")


if __name__ == "__main__":
    main()
