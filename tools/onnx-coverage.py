#!/usr/bin/env python3
"""Builds the ONNX operator coverage matrix (Markdown) from the operator schema registries and conformance reports.

Inputs:
  * the operator schemas of the installed onnx package (every domain, every since_version) and, when onnxruntime is installed, its
    com.microsoft contrib operator schemas,
  * one or more conformance reports written by tests/Sharplight.Onnx.Conformance (SHARPLIGHT_CONFORMANCE_REPORT=<file>,
    JSON lines with test, suite, provider, status, message, skip, ops, fallback) and the `<file>.supported` list of operators the
    importer accepts,
  * optional notes (tools/onnx-coverage-notes.tsv: `<domain.op>\t<note>`), e.g. known limitations.

A test that passes on an accelerator with some IR operators placed on the CPU fallback device counts as "via CPU" for its operators.

Usage (in a venv with onnx and optionally onnxruntime, never the system Python):
    python -I tools/onnx-coverage.py --report cpu.jsonl --report vulkan.jsonl [--notes tools/onnx-coverage-notes.tsv] > matrix.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import onnx
import onnx.defs

DOMAINS = ["ai.onnx", "ai.onnx.ml", "ai.onnx.preview", "ai.onnx.preview.training", "com.microsoft"]

# The com.microsoft operators that onnxruntime-exported transformer, vision and quantized models commonly use.
COMMON_CONTRIB = {
    "Attention", "MultiHeadAttention", "GroupQueryAttention", "RotaryEmbedding", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization",
    "SimplifiedLayerNormalization", "EmbedLayerNormalization", "MatMulNBits", "QuickGelu", "BiasGelu", "FastGelu", "Gelu", "BiasAdd",
    "BiasSplitGelu", "FusedMatMul", "FusedConv", "FusedGemm", "GemmFastGelu", "TransposeMatMul", "GroupNorm", "SkipGroupNorm",
    "MatMulIntegerToFloat", "DynamicQuantizeMatMul", "QGemm", "QAttention", "GatherBlockQuantized", "QuantizeLinear", "DequantizeLinear",
    "QLinearAdd", "QLinearMul", "QLinearConv", "QLinearConcat", "QLinearSigmoid", "QLinearLeakyRelu", "QLinearSoftmax",
    "QLinearGlobalAveragePool", "QLinearAveragePool", "QLinearWhere", "MoE", "QMoE", "Inverse", "Trilu", "Range", "ExpandDims", "GatherND",
    "Pad", "Unique", "CDist", "MurmurHash3", "CropAndResize", "WordConvEmbedding", "NGramRepeatBlock", "NhwcMaxPool", "ComplexMul",
    "ComplexMulConj", "MatMulInteger16", "SparseAttention", "DecoderMaskedMultiHeadAttention", "PagedAttention", "ScaledSiLU", "GatedRMSNorm",
}


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


def load_schemas():
    schemas = collections.defaultdict(list)  # (domain, op) -> [(since_version, deprecated)]
    for s in onnx.defs.get_all_schemas_with_history():
        schemas[(s.domain or "ai.onnx", s.name)].append((s.since_version, s.deprecated))
    try:
        from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

        for s in ort_state.get_all_operator_schema():
            if s.domain == "com.microsoft":
                schemas[(s.domain, s.name)].append((s.since_version, s.deprecated))
    except (ImportError, AttributeError):
        pass
    return schemas


def qualified(domain, op):
    return op if domain == "ai.onnx" else f"{domain}.{op}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", default=[], help="conformance report (JSON lines); repeat per provider")
    parser.add_argument("--notes", help="TSV of per-operator notes")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    schemas = load_schemas()
    results, supported = load_reports(args.report)
    providers = sorted({p for per in results.values() for p in per})
    accelerators = [p for p in providers if p != "cpu"]
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

    def ops_in(domain):
        return sorted(name for (d, name) in schemas if d == domain)

    def tests_for(key, p):
        return [t for t in tests_of.get(key, ()) if p in results[t]]

    def all_pass(key, p):
        tests = tests_for(key, p)
        return bool(tests) and all(results[t][p]["status"] == "Pass" for t in tests)

    def native(key, p):
        return all_pass(key, p) and all(not results[t][p].get("fallback") for t in tests_for(key, p))

    contrib = " and onnxruntime's com.microsoft contrib schemas" if any(d == "com.microsoft" for d, _ in schemas) else ""
    print(f"Operator schemas of onnx {onnx.__version__} (ai.onnx opset {onnx.defs.onnx_opset_version()}){contrib}.\n")
    print("| domain | operators | importer | " + " | ".join(f"{p}: all tests pass" for p in providers)
          + "".join(f" | {p}: without CPU fallback" for p in accelerators) + " |")
    print("|---|---|---|" + "---|" * (len(providers) + len(accelerators)))
    rows = [(d, d, ops_in(d)) for d in DOMAINS]
    rows.append(("com.microsoft", "com.microsoft (common set)", [op for op in ops_in("com.microsoft") if op in COMMON_CONTRIB]))
    for domain, label, ops in rows:
        if not ops:
            continue
        keys = [qualified(domain, op) for op in ops]
        with_tests = [k for k in keys if tests_of.get(k)]
        cells = [f"{sum(all_pass(k, p) for k in with_tests)} / {len(with_tests)} with tests" for p in providers]
        cells += [f"{sum(native(k, p) for k in with_tests)}" for p in accelerators]
        print(f"| {label} | {len(ops)} | {sum(k in supported for k in keys)} | " + " | ".join(cells) + " |")
    print()
    for suite, title in [("node", "ONNX backend node tests"), ("extra", "Extra tests (ai.onnx.ml, strings, signal, com.microsoft; onnxruntime outputs)")]:
        print(f"{title}:\n")
        print("| provider | tests | pass | pass without CPU fallback | wrong | import fail | compile fail | run fail | harness unsupported |")
        print("|---|---|---|---|---|---|---|---|---|")
        for p in providers:
            records = [per[p] for per in results.values() if p in per and per[p].get("suite", "node") == suite]
            counts = collections.Counter(r["status"] for r in records)
            total = max(len(records), 1)
            nat = sum(1 for r in records if r["status"] == "Pass" and not r.get("fallback"))
            print(f"| {p} | {len(records)} | {counts['Pass']} ({100 * counts['Pass'] / total:.1f}%) | {nat} ({100 * nat / total:.1f}%) | "
                  f"{counts['Wrong']} | {counts['ImportFail']} | {counts['CompileFail']} | {counts['RunFail']} | {counts['HarnessUnsupported']} |")
        print()
    if args.summary_only:
        return

    for domain in DOMAINS:
        names = ops_in(domain)
        if domain == "com.microsoft":
            names = [op for op in names if op in COMMON_CONTRIB or qualified(domain, op) in supported or tests_of.get(qualified(domain, op))]
        if not names:
            continue
        print(f"\n### {domain}\n")
        if domain == "com.microsoft":
            print("The common set (`COMMON_CONTRIB` in tools/onnx-coverage.py) and every other contrib operator the importer accepts.\n")
        print("| operator | versions | importer | " + " | ".join(f"{p} tests" for p in providers) + " | notes |")
        print("|---|---|---|" + "---|" * len(providers) + "---|")
        for name in names:
            key = qualified(domain, name)
            versions = sorted({v for v, _ in schemas[(domain, name)]})
            deprecated = any(dep for v, dep in schemas[(domain, name)] if v == versions[-1])
            cells = []
            for p in providers:
                tests = tests_for(key, p)
                passed = [t for t in tests if results[t][p]["status"] == "Pass"]
                via_cpu = sum(1 for t in passed if results[t][p].get("fallback"))
                cell = f"{len(passed)}/{len(tests)}" if tests else "-"
                cells.append(cell + (f" ({via_cpu} via CPU)" if via_cpu else ""))
            imported = "yes" if key in supported else ("deprecated" if deprecated else "no")
            print(f"| {name} | {', '.join(map(str, versions))} | {imported} | " + " | ".join(cells) + f" | {notes.get(key, '')} |")


if __name__ == "__main__":
    main()
