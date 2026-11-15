#!/usr/bin/env python3
"""Writes conformance tests for NonMaxSuppression, RoiAlign, Loop, runtime Range and STFT with onnxruntime outputs.

The cases randomize the attributes the ONNX backend node tests fix (box format, per-class limits, thresholds, RoiAlign coordinate
modes / pooling modes / sampling ratios, loop trip count and condition forms, signal frame lengths) and store them in the node-test
format (model.onnx + test_data_set_0/input_k.pb, output_k.pb). Expected outputs come from onnxruntime's CPU provider.

Usage (in a venv with onnx, onnxruntime and numpy, never the system Python):
    python -I tools/gen-onnx-detection-tests.py [-o ~/models/onnx/sharplight-detection-tests]

tests/Sharplight.Import.Onnx.Conformance (ExtraConformanceTests) finds the directory as one of ~/models/onnx/sharplight-*-tests.
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64
BOOL = TensorProto.BOOL


def vi(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def model(nodes, inputs, outputs, initializers=(), opset=21):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(m)
    return m


def write(root, name, m, feeds, reference=False):
    """Stores the model and one data set; outputs from onnxruntime, or from the onnx reference evaluator (float64 numpy) when asked."""
    if reference:
        results = ReferenceEvaluator(m).run(None, feeds)
    else:
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        results = sess.run(None, feeds)
    case = os.path.join(root, name)
    if os.path.exists(case):
        shutil.rmtree(case)
    data = os.path.join(case, "test_data_set_0")
    os.makedirs(data)
    with open(os.path.join(case, "model.onnx"), "wb") as f:
        f.write(m.SerializeToString())
    for k, inp in enumerate(m.graph.input):
        with open(os.path.join(data, f"input_{k}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(np.asarray(feeds[inp.name]), inp.name).SerializeToString())
    for k, (out, value) in enumerate(zip(m.graph.output, results)):
        with open(os.path.join(data, f"output_{k}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(np.asarray(value), out.name).SerializeToString())


def random_boxes(rng, batch, count, center_point_box):
    x1 = rng.uniform(0, 0.7, (batch, count))
    y1 = rng.uniform(0, 0.7, (batch, count))
    w = rng.uniform(0.1, 0.4, (batch, count))
    h = rng.uniform(0.1, 0.4, (batch, count))
    if center_point_box:
        boxes = np.stack([x1 + w / 2, y1 + h / 2, w, h], axis=-1)
    else:
        boxes = np.stack([y1, x1, y1 + h, x1 + w], axis=-1)
        flip = rng.random((batch, count)) < 0.3  # corners given in either order
        boxes[flip] = boxes[flip][:, [2, 3, 0, 1]]
    return boxes.astype(np.float32)


def nms_cases(root, rng):
    for case, (batch, classes, count, cpb, per_class, iou, score) in enumerate([
        (1, 1, 40, 0, 10, 0.5, 0.0), (1, 1, 40, 1, 10, 0.5, 0.0), (2, 3, 60, 0, 5, 0.4, 0.2), (2, 3, 60, 1, 5, 0.4, 0.2),
        (3, 2, 80, 0, 100, 0.6, 0.0), (1, 4, 50, 1, 0, 0.5, 0.0), (2, 2, 30, 0, 3, 0.0, 0.0), (1, 1, 30, 0, 30, 1.0, 0.5),
    ]):
        boxes = random_boxes(rng, batch, count, cpb)
        scores = rng.random((batch, classes, count)).astype(np.float32)
        ins = [vi("boxes", FLOAT, list(boxes.shape)), vi("scores", FLOAT, list(scores.shape))]
        inits = [numpy_helper.from_array(np.array(per_class, np.int64), "max_per_class"), numpy_helper.from_array(np.array(iou, np.float32), "iou"),
                 numpy_helper.from_array(np.array(score, np.float32), "score")]
        node = helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_per_class", "iou", "score"], ["sel"], center_point_box=cpb)
        write(root, f"det_nms_{case}", model([node], ins, [vi("sel", INT64, [None, 3])], inits), {"boxes": boxes, "scores": scores})


def roi_align_cases(root, rng):
    index = 0
    for mode in ["avg", "max"]:
        for transform in ["half_pixel", "output_half_pixel"]:
            for ratio, scale in [(0, 1.0), (2, 0.5), (3, 0.25)]:
                n, c, h, w = 2, 3, 16, 20
                x = rng.standard_normal((n, c, h, w)).astype(np.float32)
                rois = rng.uniform(0, 1, (7, 4)).astype(np.float32)
                rois[:, 2:] = rois[:, :2] + rng.uniform(0.1, 0.8, (7, 2))
                rois *= np.array([w, h, w, h], np.float32) / scale
                batch_index = rng.integers(0, n, 7).astype(np.int64)
                node = helper.make_node("RoiAlign", ["X", "rois", "bi"], ["Y"], mode=mode, output_height=4, output_width=3, sampling_ratio=ratio,
                                        spatial_scale=scale, coordinate_transformation_mode=transform)
                m = model([node], [vi("X", FLOAT, list(x.shape)), vi("rois", FLOAT, [7, 4]), vi("bi", INT64, [7])], [vi("Y", FLOAT, [7, c, 4, 3])])
                write(root, f"det_roialign_{index}", m, {"X": x, "rois": rois, "bi": batch_index})
                index += 1


def loop_body(kind):
    """Body (iter, cond, s) -> (cond, s + iter, scan = s * 2); with an early exit when s exceeds a bound."""
    nodes = [
        helper.make_node("Cast", ["iter"], ["iterf"], to=FLOAT),
        helper.make_node("Add", ["s_in", "iterf"], ["s_out"]),
        helper.make_node("Mul", ["s_out", "two"], ["scan"]),
    ]
    inits = [numpy_helper.from_array(np.array(2.0, np.float32), "two")]
    if kind == "early":
        nodes.append(helper.make_node("ReduceMax", ["s_out"], ["peak"], keepdims=0))
        nodes.append(helper.make_node("Less", ["peak", "bound"], ["cond_out"]))
        inits.append(numpy_helper.from_array(np.array(10.0, np.float32), "bound"))
    else:
        nodes.append(helper.make_node("Identity", ["cond_in"], ["cond_out"]))
    return helper.make_graph(nodes, "body", [vi("iter", INT64, []), vi("cond_in", BOOL, []), vi("s_in", FLOAT, [3])],
                             [vi("cond_out", BOOL, []), vi("s_out", FLOAT, [3]), vi("scan", FLOAT, [3])], initializer=inits)


def loop_cases(root, rng):
    s0 = np.array([0.5, 1.0, -1.5], np.float32)
    for name, kind, trip, cond in [("trip", "plain", 5, None), ("trip_cond", "plain", 4, True), ("cond_early", "early", None, True), ("trip_early", "early", 8, True)]:
        inits = []
        names = ["", "", "s0"]
        if trip is not None:
            inits.append(numpy_helper.from_array(np.array(trip, np.int64), "trip"))
            names[0] = "trip"
        if cond is not None:
            inits.append(numpy_helper.from_array(np.array(cond, np.bool_), "cond"))
            names[1] = "cond"
        node = helper.make_node("Loop", names, ["s_final", "scans"], body=loop_body(kind))
        outputs = [vi("s_final", FLOAT, [3]), vi("scans", FLOAT, [None, 3])]
        write(root, f"det_loop_{name}", model([node], [vi("s0", FLOAT, [3])], outputs, inits), {"s0": s0})


def range_cases(root, rng):
    for name, dtype, (start, limit, delta) in [("f32_up", np.float32, (0.5, 7.25, 0.75)), ("f32_down", np.float32, (9.0, -3.0, -1.5)),
                                               ("i64_up", np.int64, (2, 40, 3)), ("i64_down", np.int64, (30, -5, -4)), ("i32", np.int32, (0, 17, 2))]:
        elem = {np.float32: FLOAT, np.int64: INT64, np.int32: TensorProto.INT32}[dtype]
        node = helper.make_node("Range", ["start", "limit", "delta"], ["y"])
        ins = [vi("start", elem, []), vi("limit", elem, []), vi("delta", elem, [])]
        feeds = {k: np.array(v, dtype) for k, v in zip(["start", "limit", "delta"], [start, limit, delta])}
        write(root, f"det_range_{name}", model([node], ins, [vi("y", elem, [None])]), feeds)
    # a runtime-sized range feeding a gather (the length comes from a graph input)
    table = rng.standard_normal((12, 4)).astype(np.float32)
    nodes = [helper.make_node("Range", ["zero", "n", "one"], ["idx"]), helper.make_node("Gather", ["table", "idx"], ["rows"]),
             helper.make_node("ReduceSum", ["rows", "axes"], ["y"], keepdims=0)]
    inits = [numpy_helper.from_array(np.array(0, np.int64), "zero"), numpy_helper.from_array(np.array(1, np.int64), "one"),
             numpy_helper.from_array(table, "table"), numpy_helper.from_array(np.array([1], np.int64), "axes")]
    write(root, "det_range_runtime_gather", model(nodes, [vi("n", INT64, [])], [vi("y", FLOAT, [None])], inits), {"n": np.array(9, np.int64)})


def stft_cases(root, rng):
    index = 0
    for onesided in [1, 0]:
        for window_kind in ["hann", "none"]:
            for components, length, frame, step in [(1, 400, 64, 16), (2, 300, 48, 24), (1, 513, 128, 100)]:
                signal = rng.standard_normal((2, length, components)).astype(np.float32)
                inits = [numpy_helper.from_array(np.array(step, np.int64), "step")]
                names = ["X", "step", "", "frame_length"]
                if window_kind == "hann":
                    inits.append(numpy_helper.from_array(np.hanning(frame).astype(np.float32), "window"))
                    names = ["X", "step", "window"]
                else:
                    inits.append(numpy_helper.from_array(np.array(frame, np.int64), "frame_length"))
                frames = (length - frame) // step + 1
                bins = frame // 2 + 1 if onesided and components == 1 else frame
                node = helper.make_node("STFT", names, ["Y"], onesided=onesided)
                m = model([node], [vi("X", FLOAT, [2, length, components])], [vi("Y", FLOAT, [2, frames, bins, 2])], inits)
                write(root, f"det_stft_{index}", m, {"X": signal}, reference=True)  # onnxruntime accumulates in float
                index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/models/onnx/sharplight-detection-tests"))
    root = parser.parse_args().output
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(2024)
    nms_cases(root, rng)
    roi_align_cases(root, rng)
    loop_cases(root, rng)
    range_cases(root, rng)
    stft_cases(root, rng)
    print(f"wrote {len(os.listdir(root))} cases to {root}")


if __name__ == "__main__":
    main()
