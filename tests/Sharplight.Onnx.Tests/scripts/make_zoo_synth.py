"""Small synthetic ONNX models for ops first needed by the model zoo (written next to the other synthetic cases in ~/models/onnx/ref/synth).
Each model is run through onnxruntime for the reference outputs; consumed by ReferenceTests.Synthetic_*.
"""
import json, os
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
import onnxruntime as ort

OUT = os.path.expanduser("~/models/onnx/ref/synth")
os.makedirs(OUT, exist_ok=True)
rs = np.random.RandomState(3)


def const(name, arr):
    return nh.from_array(np.asarray(arr), name)


def save(name, nodes, inputs, outputs, inits=(), batches=(1, 3), opset=17, rtol=1e-3, ms=False, fixbatch=False, atol=2e-4):
    opsets = [h.make_opsetid("", opset)] + ([h.make_opsetid("com.microsoft", 1)] if ms else [])
    g = h.make_graph(nodes, name, [h.make_tensor_value_info(n, T.FLOAT, s) for n, s in inputs],
                     [h.make_tensor_value_info(n, T.FLOAT, None) for n in outputs], list(inits))
    m = h.make_model(g, opset_imports=opsets); m.ir_version = 8
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
    json.dump({"batches": list(batches), "inputs": [{"name": n, "shape": [(batches[0] if fixbatch else -1) if d == "N" else d for d in s]} for n, s in inputs],
               "outputs": [{"name": o} for o in outputs], "rtol": rtol, "atol": atol}, open(f"{OUT}/{name}.json", "w"))
    print("wrote", name)


# DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(scales)  (the onnxruntime dynamic-quantization pattern)
W = rs.randint(-100, 100, size=(24, 16)).astype(np.int8)
save("dyn_quant_matmul", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("MatMulInteger", ["xq", "w", "xz", "wz"], ["acc"]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
    h.make_node("Cast", ["xz"], ["xzf"], to=T.FLOAT),
], [("x", ["N", 5, 24])], ["y", "xs", "xzf"], [const("w", W), const("wz", np.int8(0)), const("ws", np.float32(0.013))], batches=(2,), fixbatch=True)

# DynamicQuantizeLinear + ConvInteger
Wc = rs.randint(-60, 60, size=(6, 3, 3, 3)).astype(np.int8)
save("dyn_quant_conv", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("ConvInteger", ["xq", "w", "xz"], ["acc"], pads=[1, 1, 1, 1]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
], [("x", ["N", 3, 8, 8])], ["y"], [const("w", Wc), const("ws", np.float32(0.02))], batches=(2,), fixbatch=True)

# QLinearAdd / QLinearGlobalAveragePool (com.microsoft)
save("qlinear_add_gap", [
    h.make_node("QuantizeLinear", ["a", "s1", "z1"], ["aq"]),
    h.make_node("QuantizeLinear", ["b", "s2", "z2"], ["bq"]),
    h.make_node("QLinearAdd", ["aq", "s1", "z1", "bq", "s2", "z2", "s3", "z3"], ["cq"], domain="com.microsoft"),
    h.make_node("QLinearGlobalAveragePool", ["cq", "s3", "z3", "s4", "z4"], ["dq"], domain="com.microsoft", channels_last=0),
    h.make_node("DequantizeLinear", ["cq", "s3", "z3"], ["c"]),
    h.make_node("DequantizeLinear", ["dq", "s4", "z4"], ["d"]),
], [("a", ["N", 4, 6, 6]), ("b", ["N", 4, 6, 6])], ["c", "d"],
    [const("s1", np.float32(0.05)), const("z1", np.uint8(120)), const("s2", np.float32(0.04)), const("z2", np.uint8(130)),
     const("s3", np.float32(0.08)), const("z3", np.uint8(125)), const("s4", np.float32(0.03)), const("z4", np.uint8(110))], ms=True, batches=(2,), atol=0.1)  # +-1 quantization step: ties round differently from onnxruntime's ratio-based kernels

# strided slices (positive steps) and a negative-step flip
save("strided_slice", [
    h.make_node("Slice", ["x", "st0", "en0", "ax0", "sp0"], ["a"]),
    h.make_node("Slice", ["x", "st1", "en1", "ax1", "sp1"], ["b"]),
    h.make_node("Slice", ["x", "st2", "en2", "ax2", "sp2"], ["c"]),
], [("x", ["N", 7, 10])], ["a", "b", "c"],
    [const("st0", np.array([0], np.int64)), const("en0", np.array([10], np.int64)), const("ax0", np.array([2], np.int64)), const("sp0", np.array([2], np.int64)),
     const("st1", np.array([1], np.int64)), const("en1", np.array([10], np.int64)), const("ax1", np.array([2], np.int64)), const("sp1", np.array([3], np.int64)),
     const("st2", np.array([6], np.int64)), const("en2", np.array([-100], np.int64)), const("ax2", np.array([1], np.int64)), const("sp2", np.array([-1], np.int64))])

# uint8 weights with an asymmetric zero point (HF "quantized" exports)
Wu = rs.randint(0, 255, size=(6, 3, 3, 3)).astype(np.uint8)
save("dyn_quant_conv_u8", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("ConvInteger", ["xq", "w", "xz", "wz"], ["acc"], pads=[1, 1, 1, 1], strides=[2, 2]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
], [("x", ["N", 3, 8, 8])], ["y"], [const("w", Wu), const("wz", np.uint8(117)), const("ws", np.float32(0.004))], batches=(2,), fixbatch=True)
Wm = rs.randint(0, 255, size=(24, 16)).astype(np.uint8)
save("dyn_quant_matmul_u8", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("MatMulInteger", ["xq", "w", "xz", "wz"], ["acc"]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
], [("x", ["N", 5, 24])], ["y"], [const("w", Wm), const("wz", np.uint8(131)), const("ws", np.float32(0.004))], batches=(2,), fixbatch=True)

# per-column weight zero points and scales (HF dynamic quantization of Linear layers)
Wp = rs.randint(0, 255, size=(24, 16)).astype(np.uint8)
save("dyn_quant_matmul_perchannel", [
    h.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
    h.make_node("MatMulInteger", ["xq", "w", "xz", "wz"], ["acc"]),
    h.make_node("Cast", ["acc"], ["accf"], to=T.FLOAT),
    h.make_node("Mul", ["xs", "ws"], ["s"]),
    h.make_node("Mul", ["accf", "s"], ["y"]),
], [("x", ["N", 5, 24])], ["y"], [const("w", Wp), const("wz", rs.randint(100, 150, size=16).astype(np.uint8)), const("ws", (rs.rand(16) * 0.01 + 0.002).astype(np.float32))], batches=(2,), fixbatch=True)
