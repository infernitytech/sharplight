"""Synthetic ONNX models for symbolic and data-dependent shapes (docs/dynamic-shapes.md), written next to the other synthetic cases in
~/models/onnx/ref/synth. Each model is run through onnxruntime for the reference outputs; consumed by ReferenceTests.Synthetic_*.
Integer outputs are cast to float so that every case has float outputs.
"""
import json, os
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
import onnxruntime as ort

OUT = os.path.expanduser("~/models/onnx/ref/synth")
os.makedirs(OUT, exist_ok=True)
rs = np.random.RandomState(11)


def const(name, arr):
    return nh.from_array(np.asarray(arr), name)


def i64(name, *values):
    return const(name, np.array(values, np.int64))


def save(name, nodes, inputs, outputs, inits=(), batches=(1, 3), opset=17, rtol=1e-3, fixbatch=False, atol=2e-4):
    g = h.make_graph(nodes, name, [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in inputs],
                     [h.make_tensor_value_info(n, T.FLOAT, None) for n in outputs], list(inits))
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)]); m.ir_version = 8
    onnx.save(m, f"{OUT}/{name}.onnx")
    sess = ort.InferenceSession(f"{OUT}/{name}.onnx", providers=["CPUExecutionProvider"])
    for b in batches:
        feeds = {}
        for k, (n, s) in enumerate(inputs):
            a = (rs.randn(*[b if d == "N" else d for d in s]) * 1.5).astype(np.float32)
            feeds[n] = a; a.tofile(f"{OUT}/{name}.b{b}.in{k}.bin")
        res = sess.run(outputs, feeds)
        for k, o in enumerate(res):
            np.ascontiguousarray(o, dtype=np.float32).tofile(f"{OUT}/{name}.b{b}.out{k}.bin")
        print("  ", name, "b", b, [o.shape for o in res])
    json.dump({"batches": list(batches), "inputs": [{"name": n, "shape": [(batches[0] if fixbatch else -1) if d == "N" else d for d in s]} for n, s in inputs],
               "outputs": [{"name": o} for o in outputs], "rtol": rtol, "atol": atol}, open(f"{OUT}/{name}.json", "w"))
    print("wrote", name)


def cast_f(src, dst):
    return h.make_node("Cast", [src], [dst], to=T.FLOAT)


# ---- data-dependent output sizes ----------------------------------------------------------------------------------------
# NonMaxSuppression (corner boxes, thresholds) and the usual (batch, class, box) column gathers.
save("dyn_nms", [
    h.make_node("Abs", ["s"], ["sa"]),
    h.make_node("Sigmoid", ["sa"], ["scores"]),
    h.make_node("NonMaxSuppression", ["boxes", "scores", "max", "iou", "thr"], ["sel"]),
    cast_f("sel", "self"),
    h.make_node("Gather", ["sel", "col2"], ["boxidx"], axis=1),
    h.make_node("Gather", ["boxes", "boxidx"], ["picked"], axis=1),
], [("boxes", [1, 40, 4]), ("s", [1, 3, 40])], ["self", "picked"],
    [i64("max", 6), const("iou", np.array([0.3], np.float32)), const("thr", np.array([0.75], np.float32)), i64("col2", 2)],
    batches=(1,), fixbatch=True)

# NonMaxSuppression with center boxes and two batches
save("dyn_nms_center", [
    h.make_node("Abs", ["b"], ["boxes"]),
    h.make_node("NonMaxSuppression", ["boxes", "scores", "max", "iou"], ["sel"], center_point_box=1),
    cast_f("sel", "self"),
], [("b", [2, 30, 4]), ("scores", [2, 2, 30])], ["self"], [i64("max", 4), const("iou", np.array([0.2], np.float32))], batches=(2,), fixbatch=True)

# NonZero (symbolic batch), transposed to [count, rank] like most exports do
save("dyn_nonzero", [
    h.make_node("Greater", ["x", "zero"], ["m"]),
    h.make_node("NonZero", ["m"], ["nz"]),
    h.make_node("Transpose", ["nz"], ["nzt"], perm=[1, 0]),
    cast_f("nzt", "y"),
], [("x", ["N", 3, 4])], ["y"], [const("zero", np.float32(0))])

# Rows of a NonZero result used as gather indices (the "boolean mask indexing" pattern)
save("dyn_nonzero_rows", [
    h.make_node("Greater", ["x", "half"], ["m"]),
    h.make_node("NonZero", ["m"], ["nz"]),
    h.make_node("Gather", ["nz", "r0"], ["rows"], axis=0),
    h.make_node("Gather", ["nz", "r1"], ["cols"], axis=0),
    h.make_node("Gather", ["x", "rows"], ["xr"], axis=0),
    cast_f("cols", "y"),
], [("x", [6, 5])], ["xr", "y"], [const("half", np.float32(0.5)), i64("r0", 0), i64("r1", 1)], batches=(1,), fixbatch=True)

# Compress along an axis and over the flattened input
save("dyn_compress", [
    h.make_node("ReduceSum", ["x", "ax1"], ["rowsum"], keepdims=0),
    h.make_node("Greater", ["rowsum", "zero"], ["keep"]),
    h.make_node("Compress", ["x", "keep"], ["y"], axis=0),
    h.make_node("Greater", ["x", "zero"], ["pos"]),
    h.make_node("Compress", ["x", "pos"], ["flat"]),
], [("x", [8, 5])], ["y", "flat"], [i64("ax1", 1), const("zero", np.float32(0))], batches=(1,), fixbatch=True)

# Unique of rounded values: values, first indices, inverse indices, counts
save("dyn_unique", [
    h.make_node("Round", ["x"], ["r"]),
    h.make_node("Unique", ["r"], ["u", "idx", "inv", "cnt"], sorted=1),
    cast_f("idx", "idxf"), cast_f("inv", "invf"), cast_f("cnt", "cntf"),
], [("x", [24])], ["u", "idxf", "invf", "cntf"], batches=(1,), fixbatch=True)

# TopK whose k depends on a data-dependent count: k = min(3, count)
save("dyn_topk_k", [
    h.make_node("Greater", ["x", "zero"], ["pos"]),
    h.make_node("Compress", ["x", "pos"], ["v"], axis=0),
    h.make_node("Shape", ["v"], ["vs"]),
    h.make_node("Min", ["vs", "three"], ["k"]),
    h.make_node("TopK", ["v", "k"], ["top", "ti"], axis=0),
    cast_f("ti", "tif"),
], [("x", [12])], ["top", "tif"], [const("zero", np.float32(0)), i64("three", 3)], batches=(1,), fixbatch=True)

# ---- symbolic dims and expressions --------------------------------------------------------------------------------------
# Range / ConstantOfShape / Expand over the batch size
save("sym_range_fill", [
    h.make_node("Shape", ["x"], ["s"]),
    h.make_node("Gather", ["s", "z"], ["n"], axis=0),
    h.make_node("Range", ["start", "n", "one"], ["r"]),
    cast_f("r", "rf"),
    h.make_node("Unsqueeze", ["rf", "ax1"], ["ru"]),
    h.make_node("Add", ["x", "ru"], ["y"]),
    h.make_node("ConstantOfShape", ["s"], ["c"], value=h.make_tensor("v", T.FLOAT, [1], [2.5])),
    h.make_node("Mul", ["x", "c"], ["y2"]),
    h.make_node("Expand", ["row", "s"], ["e"]),
    h.make_node("Add", ["x", "e"], ["y3"]),
], [("x", ["N", 4])], ["y", "y2", "y3"],
    [i64("z", 0), i64("start", 0), i64("one", 1), i64("ax1", 1), const("row", np.array([[1, 2, 3, 4]], np.float32))])

# Reshapes that merge or split the batch dim (outer dims 3*N and N), with the target computed from Shape arithmetic
save("sym_reshape_expr", [
    h.make_node("Reshape", ["x", "flat"], ["a"]),
    h.make_node("Relu", ["a"], ["ar"]),
    h.make_node("Shape", ["x"], ["s"]),
    h.make_node("Gather", ["s", "z"], ["n"], axis=0),
    h.make_node("Mul", ["n", "three"], ["n3"]),
    h.make_node("Concat", ["n3", "four"], ["t"], axis=0),
    h.make_node("Reshape", ["x", "t"], ["b"]),
    h.make_node("ReduceSum", ["b", "ax1"], ["bs"], keepdims=1),
    h.make_node("Reshape", ["bs", "back"], ["c"]),
], [("x", ["N", 3, 4])], ["ar", "c"],
    [i64("flat", -1, 4), i64("z", 0), i64("three", 3), i64("four", 4), i64("ax1", 1), i64("back", -1, 3)])

# DynamicQuantizeLinear with a symbolic batch
W = rs.randint(-100, 100, size=(24, 16)).astype(np.int8)
save("sym_dyn_quant", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("MatMulInteger", ["xq", "w", "xz", "wz"], ["acc"]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
], [("x", ["N", 5, 24])], ["y"], [const("w", W), const("wz", np.int8(0)), const("ws", np.float32(0.013))], atol=2e-3)

# ---- control flow ------------------------------------------------------------------------------------------------------
# Loop with a constant body (TF arange style): evaluated at import time
body = h.make_graph([
    h.make_node("Identity", ["cond"], ["cond_out"]),
    h.make_node("Add", ["prev", "step"], ["cur"]),
    h.make_node("Identity", ["prev"], ["range"]),
], "arange", [h.make_tensor_value_info("i", T.INT64, []), h.make_tensor_value_info("cond", T.BOOL, []), h.make_tensor_value_info("prev", T.FLOAT, [])],
    [h.make_tensor_value_info("cond_out", T.BOOL, []), h.make_tensor_value_info("cur", T.FLOAT, []), h.make_tensor_value_info("range", T.FLOAT, [])],
    [const("step", np.float32(0.5))])
save("loop_fold", [
    h.make_node("Loop", ["trip", "true", "start"], ["last", "seq"], body=body),
    h.make_node("Add", ["x", "seq"], ["y"]),
], [("x", ["N", 6])], ["y"], [i64("trip", 6), const("true", np.array(True)), const("start", np.float32(1))])

# Loop whose body depends on runtime tensors: unrolled (carried state + scan output)
body2 = h.make_graph([
    h.make_node("Identity", ["cond"], ["cond_out"]),
    h.make_node("Mul", ["acc", "decay"], ["d"]),
    h.make_node("Add", ["d", "x"], ["acc_out"]),
    h.make_node("Tanh", ["acc_out"], ["scan"]),
], "ema", [h.make_tensor_value_info("i", T.INT64, []), h.make_tensor_value_info("cond", T.BOOL, []), h.make_tensor_value_info("acc", T.FLOAT, [2, 5])],
    [h.make_tensor_value_info("cond_out", T.BOOL, []), h.make_tensor_value_info("acc_out", T.FLOAT, [2, 5]), h.make_tensor_value_info("scan", T.FLOAT, [2, 5])],
    [const("decay", np.float32(0.7))])
save("loop_unroll", [
    h.make_node("Loop", ["trip", "", "x"], ["final", "steps"], body=body2),
], [("x", [2, 5])], ["final", "steps"], [i64("trip", 4)], batches=(1,), fixbatch=True)

# Scan: running sum over the rows of x, forward and reverse
sbody = h.make_graph([
    h.make_node("Add", ["state", "row"], ["state_out"]),
    h.make_node("Identity", ["state_out"], ["out"]),
], "cumsum", [h.make_tensor_value_info("state", T.FLOAT, [4]), h.make_tensor_value_info("row", T.FLOAT, [4])],
    [h.make_tensor_value_info("state_out", T.FLOAT, [4]), h.make_tensor_value_info("out", T.FLOAT, [4])])
save("scan_cumsum", [
    h.make_node("Scan", ["init", "x"], ["total", "partial"], body=sbody, num_scan_inputs=1, scan_input_directions=[1]),
], [("init", [4]), ("x", [5, 4])], ["total", "partial"], batches=(1,), fixbatch=True)
