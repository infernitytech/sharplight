#!/usr/bin/env python3
"""Writes the ONNX backend *node* conformance tests (model.onnx + test_data_set_*/input_*.pb, output_*.pb).

The onnx wheels no longer ship onnx/backend/test/data/node, and `backend-test-tools generate-data` only writes the
model tests, so this script serializes the node test cases of the installed onnx package the same way.

Usage (in a venv with onnx and numpy, never the system Python):
    python -I tools/gen-onnx-node-tests.py [-o ~/models/onnx/backend-node-<onnx version>]

tests/Sharplight.Onnx.Conformance reads the directory from SHARPLIGHT_ONNX_NODE_TESTS, else the newest
~/models/onnx/backend-node-* directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings

import onnx
from onnx import TensorProto, numpy_helper
import onnx.backend.test.case.node as node_cases


def serialize(value, type_proto, name: str) -> bytes:
    if type_proto.HasField("map_type"):
        return numpy_helper.from_dict(value, name).SerializeToString()
    if type_proto.HasField("sequence_type"):
        return numpy_helper.from_list(value, name).SerializeToString()
    if type_proto.HasField("optional_type"):
        return numpy_helper.from_optional(value, name).SerializeToString()
    if isinstance(value, TensorProto):
        return value.SerializeToString()
    return numpy_helper.from_array(value, name).SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = os.path.expanduser(f"~/models/onnx/backend-node-{onnx.__version__}")
    parser.add_argument("-o", "--output", default=default, help="output directory (default: %(default)s)")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    cases = node_cases.collect_testcases(None)
    os.makedirs(args.output, exist_ok=True)
    for case in cases:
        case_dir = os.path.join(args.output, case.name)
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
        os.makedirs(case_dir)
        with open(os.path.join(case_dir, "model.onnx"), "wb") as f:
            f.write(case.model.SerializeToString())
        for i, (inputs, outputs) in enumerate(case.data_sets):
            set_dir = os.path.join(case_dir, f"test_data_set_{i}")
            os.makedirs(set_dir)
            graph = case.model.graph
            for j, value in enumerate(inputs):
                with open(os.path.join(set_dir, f"input_{j}.pb"), "wb") as f:
                    f.write(serialize(value, graph.input[j].type, graph.input[j].name))
            for j, value in enumerate(outputs):
                with open(os.path.join(set_dir, f"output_{j}.pb"), "wb") as f:
                    f.write(serialize(value, graph.output[j].type, graph.output[j].name))
    print(f"{len(cases)} node tests written to {args.output} (onnx {onnx.__version__})")


if __name__ == "__main__":
    main()
