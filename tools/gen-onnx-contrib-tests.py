#!/usr/bin/env python3
"""Writes conformance tests for the com.microsoft contrib operators that onnxruntime-exported models use, with onnxruntime outputs.

The ONNX backend node tests do not cover the contrib domain. This script builds small models in the layouts the onnxruntime
transformer optimizer, onnxruntime-genai and the quantization tools emit, and stores them in the node-test format
(model.onnx + test_data_set_0/input_k.pb, output_k.pb). The expected outputs are computed by onnxruntime's CPU provider, the
reference implementation of these operators; cases the CPU provider cannot run are reported and skipped.

Usage (in a venv with onnx, onnxruntime and numpy, never the system Python):
    python -I tools/gen-onnx-contrib-tests.py [-o ~/models/onnx/sharplight-contrib-tests]

tests/Sharplight.Onnx.Conformance (ExtraConformanceTests) finds the directory as one of ~/models/onnx/sharplight-*-tests.
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

MS = "com.microsoft"
RNG = np.random.default_rng(1234)

TYPES = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.int16): TensorProto.INT16,
    np.dtype(np.uint32): TensorProto.UINT32,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.bool_): TensorProto.BOOL,
}


def f32(*shape, scale=1.0):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def u8(*shape):
    return RNG.integers(0, 256, shape, dtype=np.uint8)


def i8(*shape):
    return RNG.integers(-128, 128, shape, dtype=np.int8)


def ref_fast_gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def ref_gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def ref_bias_split_gelu(v):
    y = v["x"] + v["b"]
    left, right = np.split(y, 2, axis=-1)
    return [left * ref_gelu(right)]


def ref_group_norm(x, gamma, beta, groups, silu, eps=1e-5):
    n, h, w, c = x.shape
    g = x.reshape(n, h * w, groups, c // groups).astype(np.float64)
    mean = g.mean(axis=(1, 3), keepdims=True)
    var = g.var(axis=(1, 3), keepdims=True)
    y = ((g - mean) / np.sqrt(var + eps)).reshape(n, h, w, c) * gamma + beta
    return y / (1 + np.exp(-y)) if silu else y


def ref_complex_mul(a, b, conj):
    a, b = np.broadcast_arrays(a, b)
    ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], (-b[..., 1] if conj else b[..., 1])
    return np.stack([ar * br - ai * bi, ar * bi + ai * br], axis=-1)


class Case:
    def __init__(self, name, op, inputs, outputs, attrs=None, inits=None, domain=MS, opset=21, extra_nodes=(), ref=None):
        """inputs: list of (name, array or None for an absent optional input); inits: {name: array} (constant inputs); ref: a numpy
        reference (inputs and constants by name -> outputs) for operators only onnxruntime's GPU providers implement."""
        self.name, self.op, self.inputs, self.outputs = name, op, inputs, outputs
        self.attrs, self.inits, self.domain, self.opset = attrs or {}, inits or {}, domain, opset
        self.extra_nodes = extra_nodes
        self.ref = ref

    def model(self):
        inits = [numpy_helper.from_array(v, k) for k, v in self.inits.items()]
        node_inputs = [n if (v is not None or n in self.inits) else "" for n, v in self.inputs]
        node = helper.make_node(self.op, node_inputs, self.outputs, domain=self.domain, **self.attrs)
        graph_inputs = [helper.make_tensor_value_info(n, TYPES[v.dtype], v.shape) for n, v in self.inputs if v is not None]
        graph_outputs = [helper.make_empty_tensor_value_info(o) for o in self.outputs if o]
        graph = helper.make_graph([node, *self.extra_nodes], self.name, graph_inputs, graph_outputs, initializer=inits)
        opsets = [helper.make_opsetid("", self.opset), helper.make_opsetid(MS, 1)]
        return helper.make_model(graph, opset_imports=opsets, ir_version=10)

    def feeds(self):
        return {n: v for n, v in self.inputs if v is not None}


def fix_output_types(m, results):
    for vi, r in zip(m.graph.output, results):
        vi.type.CopyFrom(helper.make_tensor_type_proto(TYPES[np.asarray(r).dtype], np.asarray(r).shape))


def cases():
    B, S, N, H = 2, 5, 4, 8
    D = N * H
    out = []

    # ---- activations -----------------------------------------------------------------------------------------------
    x = f32(B, S, D)
    bias = f32(D)
    out += [
        Case("ms_gelu", "Gelu", [("x", x)], ["y"]),
        Case("ms_bias_gelu", "BiasGelu", [("x", x), ("b", bias)], ["y"]),
        Case("ms_fast_gelu", "FastGelu", [("x", x)], ["y"]),
        Case("ms_fast_gelu_bias", "FastGelu", [("x", x), ("b", bias)], ["y"]),
        Case("ms_quick_gelu", "QuickGelu", [("x", x)], ["y"]),
        Case("ms_quick_gelu_alpha", "QuickGelu", [("x", x)], ["y"], {"alpha": 1.0}),
        Case("ms_bias_add", "BiasAdd", [("x", f32(B, 6, D)), ("b", bias), ("skip", f32(B, 6, D))], ["y"], ref=lambda v: [v["x"] + v["b"] + v["skip"]]),
        Case("ms_bias_split_gelu", "BiasSplitGelu", [("x", f32(B, 6, 2 * D)), ("b", f32(2 * D))], ["y"], ref=ref_bias_split_gelu),
        Case("ms_scaled_silu", "ScaledSiLU", [("x", x)], ["y"], {"alpha": 0.5}),
        Case("ms_bias_softmax", "BiasSoftmax", [("x", f32(B, N, S, S)), ("b", f32(B, 1, S, S))], ["y"], {"axis": 3, "is_inner_broadcast": 0}),
        Case("ms_bias_dropout", "BiasDropout", [("x", x), ("b", bias), ("r", f32(B, S, D))], ["y"]),
    ]

    # ---- normalization ---------------------------------------------------------------------------------------------
    gamma, beta = f32(D), f32(D)
    out += [
        Case("ms_skip_layer_norm", "SkipLayerNormalization", [("x", x), ("skip", f32(B, S, D)), ("gamma", gamma), ("beta", beta)], ["y"],
             {"epsilon": 1e-5}),
        Case("ms_skip_layer_norm_bias_outputs", "SkipLayerNormalization",
             [("x", x), ("skip", f32(B, S, D)), ("gamma", gamma), ("beta", beta), ("bias", bias)], ["y", "mean", "inv_std", "sum"],
             {"epsilon": 1e-5}),
        Case("ms_skip_layer_norm_broadcast_skip", "SkipLayerNormalization", [("x", x), ("skip", f32(1, S, D)), ("gamma", gamma)], ["y"]),
        Case("ms_skip_simplified_layer_norm", "SkipSimplifiedLayerNormalization", [("x", x), ("skip", f32(B, S, D)), ("gamma", gamma)], ["y"],
             {"epsilon": 1e-6}),
        Case("ms_skip_simplified_layer_norm_sum", "SkipSimplifiedLayerNormalization",
             [("x", x), ("skip", f32(B, S, D)), ("gamma", gamma), ("bias", bias)], ["y", "", "", "sum"]),
        Case("ms_simplified_layer_norm", "SimplifiedLayerNormalization", [("x", x), ("gamma", gamma)], ["y"], {"axis": -1, "epsilon": 1e-6},
             domain=""),
        Case("ms_simplified_layer_norm_inv_std", "SimplifiedLayerNormalization", [("x", x), ("gamma", gamma)], ["y", "inv_std"],
             {"axis": -1, "epsilon": 1e-6}, domain=""),
        Case("ms_layer_norm_contrib_domain", "LayerNormalization", [("x", x), ("gamma", gamma), ("beta", beta)], ["y"], {"axis": -1}, domain=""),
        Case("ms_group_norm_nhwc", "GroupNorm", [("x", f32(2, 4, 4, 32)), ("gamma", f32(32)), ("beta", f32(32))], ["y"],
             {"groups": 4, "epsilon": 1e-5, "activation": 0, "channels_last": 1}, ref=lambda v: [ref_group_norm(v["x"], v["gamma"], v["beta"], 4, False)]),
        Case("ms_group_norm_nhwc_silu", "GroupNorm", [("x", f32(2, 4, 4, 32)), ("gamma", f32(32)), ("beta", f32(32))], ["y"],
             {"groups": 8, "epsilon": 1e-5, "activation": 1, "channels_last": 1}, ref=lambda v: [ref_group_norm(v["x"], v["gamma"], v["beta"], 8, True)]),
        Case("ms_group_norm_nchw", "GroupNorm", [("x", f32(2, 32, 4, 4)), ("gamma", f32(32)), ("beta", f32(32))], ["y"],
             {"groups": 4, "epsilon": 1e-5, "activation": 0, "channels_last": 0},
             ref=lambda v: [ref_group_norm(v["x"].transpose(0, 2, 3, 1), v["gamma"], v["beta"], 4, False).transpose(0, 3, 1, 2)]),
        Case("ms_skip_group_norm", "SkipGroupNorm", [("x", f32(2, 4, 4, 32)), ("gamma", f32(32)), ("beta", f32(32)), ("skip", f32(2, 1, 1, 32)),
                                                     ("bias", f32(32))], ["y", "s"], {"groups": 4, "epsilon": 1e-5, "activation": 1, "channels_last": 1},
             ref=lambda v: [ref_group_norm(v["x"] + v["skip"] + v["bias"], v["gamma"], v["beta"], 4, True), v["x"] + v["skip"] + v["bias"]]),
        Case("ms_gated_rms_norm", "GatedRMSNorm", [("x", x), ("scale", gamma), ("gate", f32(B, S, D))], ["y"], {"epsilon": 1e-6, "activation": "silu"}),
    ]
    vocab, maxpos, types = 30, 16, 2
    ids = RNG.integers(0, vocab, (B, S)).astype(np.int32)
    seg = RNG.integers(0, types, (B, S)).astype(np.int32)
    mask = np.ones((B, S), np.int32)
    mask[1, 3:] = 0
    emb = {"word": f32(vocab, D), "pos": f32(maxpos, D), "seg": f32(types, D)}
    out += [
        Case("ms_embed_layer_norm", "EmbedLayerNormalization",
             [("ids", ids), ("seg_ids", seg), ("word", None), ("pos", None), ("segemb", None), ("gamma", None), ("beta", None), ("mask", mask)],
             ["y", "mask_index"], {"epsilon": 1e-12},
             inits={"word": emb["word"], "pos": emb["pos"], "segemb": emb["seg"], "gamma": gamma, "beta": beta}),
        Case("ms_embed_layer_norm_sum", "EmbedLayerNormalization",
             [("ids", ids), ("seg_ids", None), ("word", None), ("pos", None), ("segemb", None), ("gamma", None), ("beta", None)],
             ["y", "", "sum"], {"epsilon": 1e-12},
             inits={"word": emb["word"], "pos": emb["pos"], "gamma": gamma, "beta": beta}),
    ]

    # ---- linear ----------------------------------------------------------------------------------------------------
    out += [
        Case("ms_fused_matmul", "FusedMatMul", [("a", f32(B, S, D)), ("b", f32(D, 6))], ["y"], {"alpha": 0.5}),
        Case("ms_fused_matmul_trans", "FusedMatMul", [("a", f32(B, D, S)), ("b", f32(B, 6, D))], ["y"], {"transA": 1, "transB": 1}),
        Case("ms_fused_matmul_trans_batch", "FusedMatMul", [("a", f32(3, B, S)), ("b", f32(B, 3, 6))], ["y"], {"transBatchA": 1, "transA": 1, "transB": 0}),
        Case("ms_transpose_matmul", "TransposeMatMul", [("a", f32(D, S)), ("b", f32(D, 6))], ["y"], {"transA": 1}),
        Case("ms_fused_gemm_relu", "FusedGemm", [("a", f32(S, D)), ("b", f32(D, 6)), ("c", f32(6))], ["y"], {"activation": "Relu"}),
        Case("ms_fused_gemm_leaky", "FusedGemm", [("a", f32(D, S)), ("b", f32(6, D)), ("c", f32(6))], ["y"],
             {"activation": "LeakyRelu", "activation_alpha": 0.1, "transA": 1, "transB": 1, "alpha": 0.7, "beta": 0.3}),
        Case("ms_gemm_fast_gelu", "GemmFastGelu", [("a", f32(B, S, D)), ("w", None), ("b", None)], ["y"], inits={"w": f32(D, 6), "b": f32(6)},
             ref=lambda v: [ref_fast_gelu(v["a"] @ v["w"] + v["b"])]),
        Case("ms_fused_conv_relu", "FusedConv", [("x", f32(1, 3, 8, 8)), ("w", None), ("b", None)], ["y"],
             {"activation": "Relu", "kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, inits={"w": f32(4, 3, 3, 3), "b": f32(4)}),
        Case("ms_fused_conv_sum_sigmoid", "FusedConv", [("x", f32(1, 3, 8, 8)), ("w", None), ("b", None), ("z", f32(1, 4, 6, 6))], ["y"],
             {"activation": "Sigmoid", "kernel_shape": [3, 3]}, inits={"w": f32(4, 3, 3, 3), "b": f32(4)}),
        Case("ms_fused_conv_clip", "FusedConv", [("x", f32(1, 3, 8, 8)), ("w", None)], ["y"],
             {"activation": "Clip", "activation_params": [-0.5, 0.5], "kernel_shape": [3, 3]}, inits={"w": f32(4, 3, 3, 3)}),
    ]

    # ---- MatMulNBits ------------------------------------------------------------------------------------------------
    for bits, block, K, Nout, zp, with_bias, name in [
        (4, 32, 64, 24, False, False, "ms_matmul_nbits_4b"),
        (4, 32, 64, 24, True, True, "ms_matmul_nbits_4b_zp_bias"),
        (4, 16, 40, 8, True, False, "ms_matmul_nbits_4b_ragged_k"),
        (8, 32, 64, 24, False, False, "ms_matmul_nbits_8b"),
        (8, 64, 128, 16, True, False, "ms_matmul_nbits_8b_zp"),
        (2, 32, 64, 8, True, False, "ms_matmul_nbits_2b_zp"),
    ]:
        n_blocks = (K + block - 1) // block
        blob = block * bits // 8
        packed = u8(Nout, n_blocks, blob)
        scales = (np.abs(f32(Nout * n_blocks)) * 0.1 + 0.01).astype(np.float32)
        inits = {"B": packed, "scales": scales}
        ins = [("A", f32(B, 3, K)), ("B", None), ("scales", None)]
        if zp:
            zp_bytes = (n_blocks * bits + 7) // 8
            inits["zp"] = u8(Nout, zp_bytes).reshape(-1)
            ins.append(("zp", None))
        if with_bias:
            if not zp:
                ins.append(("zp", None))
            ins += [("g_idx", None), ("bias", None)]
            inits["bias"] = f32(Nout)
        out.append(Case(name, "MatMulNBits", ins, ["y"], {"K": K, "N": Nout, "bits": bits, "block_size": block}, inits=inits))

    # ---- quantized linear --------------------------------------------------------------------------------------------
    a_q, b_q = u8(3, 16), i8(16, 6)
    out += [
        Case("ms_matmul_integer_to_float", "MatMulIntegerToFloat",
             [("a", a_q), ("b", None), ("as", None), ("bs", None), ("az", None), ("bz", None), ("bias", None)], ["y"],
             inits={"b": b_q, "as": np.array(0.02, np.float32), "bs": (np.abs(f32(6)) * 0.01).astype(np.float32), "az": np.array(128, np.uint8),
                    "bz": np.zeros(6, np.int8), "bias": f32(6)}),
        Case("ms_dynamic_quantize_matmul", "DynamicQuantizeMatMul", [("a", f32(3, 16)), ("b", None), ("bs", None), ("bz", None), ("bias", None)], ["y"],
             inits={"b": b_q, "bs": np.array(0.01, np.float32), "bz": np.array(0, np.int8), "bias": f32(6)}),
        Case("ms_qgemm", "QGemm", [("a", a_q), ("as", None), ("az", None), ("b", None), ("bs", None), ("bz", None), ("c", None)], ["y"],
             {"alpha": 1.0}, inits={"as": np.array(0.02, np.float32), "az": np.array(120, np.uint8), "b": b_q.astype(np.uint8),
                                    "bs": np.array(0.01, np.float32), "bz": np.array(128, np.uint8), "c": RNG.integers(-50, 50, (6,), dtype=np.int32)}),
        Case("ms_gather_block_quantized_4b", "GatherBlockQuantized", [("data", None), ("idx", np.array([[0, 3], [5, 1]], np.int64)), ("scales", None)],
             ["y"], {"block_size": 16, "bits": 4, "gather_axis": 0, "quantize_axis": 1},
             inits={"data": u8(6, 16), "scales": (np.abs(f32(6, 2)) + 0.1).astype(np.float32)}),
        Case("ms_gather_block_quantized_4b_zp", "GatherBlockQuantized",
             [("data", None), ("idx", np.array([2, 4], np.int64)), ("scales", None), ("zp", None)], ["y"],
             {"block_size": 16, "bits": 4, "gather_axis": 0, "quantize_axis": 1},
             inits={"data": u8(6, 16), "scales": (np.abs(f32(6, 2)) + 0.1).astype(np.float32), "zp": u8(6, 1)}),
        Case("ms_gather_block_quantized_8b", "GatherBlockQuantized", [("data", None), ("idx", np.array([1, 0, 5], np.int64)), ("scales", None)],
             ["y"], {"block_size": 16, "bits": 8, "gather_axis": 0, "quantize_axis": 1},
             inits={"data": u8(6, 32), "scales": (np.abs(f32(6, 2)) + 0.1).astype(np.float32)}),
        Case("ms_quantize_linear", "QuantizeLinear", [("x", f32(B, S, 6)), ("s", None), ("z", None)], ["y"],
             inits={"s": np.array(0.05, np.float32), "z": np.array(3, np.uint8)}),
        Case("ms_dequantize_linear", "DequantizeLinear", [("x", u8(B, S, 6)), ("s", None), ("z", None)], ["y"],
             inits={"s": np.array(0.05, np.float32), "z": np.array(3, np.uint8)}),
        Case("ms_dequantize_linear_axis", "DequantizeLinear", [("x", i8(B, 6, 3)), ("s", None), ("z", None)], ["y"], {"axis": 1},
             inits={"s": (np.abs(f32(6)) * 0.1).astype(np.float32), "z": np.zeros(6, np.int8)}),
        Case("ms_qlinear_where", "QLinearWhere",
             [("c", RNG.random((B, 6)) > 0.5), ("x", u8(B, 6)), ("xs", None), ("xz", None), ("y", u8(B, 6)), ("ys", None), ("yz", None),
              ("zs", None), ("zz", None)], ["z"],
             inits={"xs": np.array(0.1, np.float32), "xz": np.array(10, np.uint8), "ys": np.array(0.05, np.float32), "yz": np.array(100, np.uint8),
                    "zs": np.array(0.08, np.float32), "zz": np.array(50, np.uint8)}),
        Case("ms_qlinear_reduce_mean", "QLinearReduceMean", [("x", u8(B, 6, 4)), ("xs", None), ("xz", None), ("ys", None), ("yz", None)], ["y"],
             {"axes": [2], "keepdims": 1},
             inits={"xs": np.array(0.1, np.float32), "xz": np.array(10, np.uint8), "ys": np.array(0.05, np.float32), "yz": np.array(20, np.uint8)}),
        Case("ms_matmul_integer16", "MatMulInteger16", [("a", RNG.integers(-300, 300, (3, 8), dtype=np.int16)),
                                                         ("b", RNG.integers(-300, 300, (8, 4), dtype=np.int16))], ["y"]),
    ]

    # ---- attention -------------------------------------------------------------------------------------------------
    w_qkv, b_qkv = f32(D, 3 * D, scale=0.3), f32(3 * D)
    mask_len = np.array([S, 3], np.int32)
    raw_mask = np.ones((B, S), np.int32)
    raw_mask[1, 3:] = 0
    P = 3
    out += [
        Case("ms_attention", "Attention", [("x", x), ("w", None), ("b", None)], ["y"], {"num_heads": N}, inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_unidirectional", "Attention", [("x", x), ("w", None), ("b", None)], ["y"], {"num_heads": N, "unidirectional": 1},
             inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_mask_index", "Attention", [("x", x), ("w", None), ("b", None), ("m", mask_len)], ["y"], {"num_heads": N},
             inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_raw_mask", "Attention", [("x", x), ("w", None), ("b", None), ("m", raw_mask)], ["y"], {"num_heads": N},
             inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_mask_3d", "Attention", [("x", x), ("w", None), ("b", None), ("m", np.tril(np.ones((B, S, S), np.int32)))], ["y"],
             {"num_heads": N}, inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_past", "Attention", [("x", f32(B, 1, D)), ("w", None), ("b", None), ("m", np.ones((B, P + 1), np.int32)),
                                                 ("past", f32(2, B, N, P, H))], ["y", "present"], {"num_heads": N, "unidirectional": 1},
             inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_bias", "Attention", [("x", x), ("w", None), ("b", None), ("m", None), ("past", None), ("ab", f32(1, N, S, S))], ["y"],
             {"num_heads": N}, inits={"w": w_qkv, "b": b_qkv}),
        Case("ms_attention_qkv_sizes", "Attention", [("x", x), ("w", None), ("b", None)], ["y"],
             {"num_heads": N, "qkv_hidden_sizes": [D, D, 2 * D]}, inits={"w": f32(D, 4 * D, scale=0.3), "b": f32(4 * D)}),
        Case("ms_attention_scale", "Attention", [("x", x), ("w", None), ("b", None)], ["y"], {"num_heads": N, "scale": 0.2},
             inits={"w": w_qkv, "b": b_qkv}),
    ]
    q, k, v = f32(B, S, D), f32(B, 7, D), f32(B, 7, D)
    out += [
        Case("ms_mha", "MultiHeadAttention", [("q", q), ("k", k), ("v", v)], ["y"], {"num_heads": N}),
        Case("ms_mha_bias", "MultiHeadAttention", [("q", q), ("k", k), ("v", v), ("b", f32(3 * D))], ["y"], {"num_heads": N}),
        Case("ms_mha_key_padding_mask", "MultiHeadAttention",
             [("q", q), ("k", k), ("v", v), ("b", None), ("m", np.array([[1] * 7, [1] * 4 + [0] * 3], np.int32))], ["y"], {"num_heads": N}),
        Case("ms_mha_key_lengths", "MultiHeadAttention", [("q", q), ("k", k), ("v", v), ("b", None), ("m", np.array([7, 4], np.int32))], ["y"],
             {"num_heads": N}),
        Case("ms_mha_attention_bias", "MultiHeadAttention", [("q", q), ("k", k), ("v", v), ("b", None), ("m", None), ("ab", f32(B, N, S, 7))], ["y"],
             {"num_heads": N}),
        Case("ms_mha_packed_qkv", "MultiHeadAttention", [("q", f32(B, S, N, 3, H))], ["y"], {"num_heads": N}),
        Case("ms_mha_packed_kv", "MultiHeadAttention", [("q", q), ("k", f32(B, 7, N, 2, H))], ["y"], {"num_heads": N}),
        Case("ms_mha_bnsh_kv", "MultiHeadAttention", [("q", q), ("k", f32(B, N, 7, H)), ("v", f32(B, N, 7, H))], ["y"], {"num_heads": N}),
        Case("ms_mha_past", "MultiHeadAttention",
             [("q", f32(B, 2, D)), ("k", f32(B, 2, D)), ("v", f32(B, 2, D)), ("b", None), ("m", None), ("ab", None),
              ("pk", f32(B, N, P, H)), ("pv", f32(B, N, P, H))], ["y", "present_k", "present_v"], {"num_heads": N}),
        Case("ms_mha_unidirectional", "MultiHeadAttention", [("q", q), ("k", f32(B, S, D)), ("v", f32(B, S, D))], ["y"],
             {"num_heads": N, "unidirectional": 1}),
        Case("ms_mha_scale", "MultiHeadAttention", [("q", q), ("k", k), ("v", v)], ["y"], {"num_heads": N, "scale": 0.3}),
    ]

    # GroupQueryAttention: prompt, decode with a shared past buffer, rotary, local window, softcap, packed QKV.
    Nq, Nkv = 4, 2
    half = H // 2
    pos = np.arange(32, dtype=np.float32)[:, None] / (10000 ** (np.arange(half, dtype=np.float32) / half))[None, :]
    cos, sin = np.cos(pos).astype(np.float32), np.sin(pos).astype(np.float32)
    pos16 = np.arange(32, dtype=np.float32)[:, None] / (10000 ** (np.arange(8, dtype=np.float32) / 8))[None, :]
    cos16, sin16 = np.cos(pos16).astype(np.float32), np.sin(pos16).astype(np.float32)

    def gqa(name, s, past, total, attrs=None, packed=False, rotary=False, seqlens=None, batch=B):
        H = 16 if rotary else 8
        B = batch
        ins = []
        if packed:
            ins += [("q", f32(B, s, (Nq + 2 * Nkv) * H)), ("k", None), ("v", None)]
        else:
            ins += [("q", f32(B, s, Nq * H)), ("k", f32(B, s, Nkv * H)), ("v", f32(B, s, Nkv * H))]
        if past:
            ins += [("pk", f32(B, Nkv, past, H)), ("pv", f32(B, Nkv, past, H))]
        else:
            ins += [("pk", None), ("pv", None)]
        lens = seqlens if seqlens is not None else np.full(B, total - 1, np.int32)
        ins += [("seqlens_k", lens.astype(np.int32)), ("total", np.array(total, np.int32))]
        inits = {}
        if rotary:
            ins += [("cos", None), ("sin", None)]
            inits = {"cos": cos16, "sin": sin16}
        a = {"num_heads": Nq, "kv_num_heads": Nkv}
        a.update(attrs or {})
        if rotary:
            a["do_rotary"] = 1
        return Case(name, "GroupQueryAttention", ins, ["y", "present_k", "present_v"], a, inits=inits)

    out += [
        gqa("ms_gqa_prompt", S, 0, S),
        gqa("ms_gqa_prompt_packed", S, 0, S, packed=True),
        gqa("ms_gqa_prompt_rotary", S, 0, S, rotary=True),
        gqa("ms_gqa_prompt_rotary_interleaved", S, 0, S, {"rotary_interleaved": 1}, rotary=True),
        gqa("ms_gqa_prompt_local_window", 6, 0, 6, {"local_window_size": 2}),
        gqa("ms_gqa_prompt_softcap", S, 0, S, {"softcap": 2.0}),
        gqa("ms_gqa_prompt_scale", S, 0, S, {"scale": 0.2}),
        gqa("ms_gqa_prompt_smooth_softmax", S, 0, S, {"smooth_softmax": 1}),
        gqa("ms_gqa_decode", 1, 8, 6, seqlens=np.array([5, 5])),
        gqa("ms_gqa_decode_rotary", 1, 8, 6, rotary=True, seqlens=np.array([5, 5])),
        gqa("ms_gqa_decode_ragged", 1, 8, 6, seqlens=np.array([5, 3])),
        gqa("ms_gqa_append", 2, 4, 6, seqlens=np.array([5]), batch=1),
        gqa("ms_gqa_append_rotary", 3, 8, 7, seqlens=np.array([6]), batch=1, rotary=True),
    ]

    # RotaryEmbedding (contrib): 3-D and 4-D input, position ids or an offset, interleaved, partial rotary.
    rot = Case
    out += [
        rot("ms_rotary_embedding", "RotaryEmbedding", [("x", f32(B, S, D)), ("pos", np.tile(np.arange(S, dtype=np.int64), (B, 1))), ("cos", None), ("sin", None)],
            ["y"], {"num_heads": N}, inits={"cos": cos, "sin": sin}),
        rot("ms_rotary_embedding_offset", "RotaryEmbedding", [("x", f32(B, S, D)), ("pos", np.array([3], np.int64)), ("cos", None), ("sin", None)],
            ["y"], {"num_heads": N}, inits={"cos": cos, "sin": sin}),
        rot("ms_rotary_embedding_interleaved", "RotaryEmbedding",
            [("x", f32(B, S, D)), ("pos", np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]], np.int64)), ("cos", None), ("sin", None)],
            ["y"], {"num_heads": N, "interleaved": 1}, inits={"cos": cos, "sin": sin}),
        rot("ms_rotary_embedding_bnsh", "RotaryEmbedding", [("x", f32(B, N, S, H)), ("pos", np.tile(np.arange(S, dtype=np.int64), (B, 1))), ("cos", None), ("sin", None)],
            ["y"], {}, inits={"cos": cos, "sin": sin}),
        rot("ms_rotary_embedding_partial", "RotaryEmbedding",
            [("x", f32(B, S, N * 16)), ("pos", np.tile(np.arange(S, dtype=np.int64), (B, 1))), ("cos", None), ("sin", None)],
            ["y"], {"num_heads": N, "rotary_embedding_dim": 8}, inits={"cos": cos[:, :4].copy(), "sin": sin[:, :4].copy()}),
    ]

    # ---- miscellaneous ------------------------------------------------------------------------------------------------
    sq = f32(3, 4, 4)
    out += [
        Case("ms_inverse", "Inverse", [("x", (sq + np.eye(4, dtype=np.float32) * 4).astype(np.float32))], ["y"]),
        Case("ms_cdist_sqeuclidean", "CDist", [("a", f32(5, 3)), ("b", f32(4, 3))], ["y"], {"metric": "sqeuclidean"}),
        Case("ms_cdist_euclidean", "CDist", [("a", f32(5, 3)), ("b", f32(4, 3))], ["y"], {"metric": "euclidean"}),
        Case("ms_expand_dims", "ExpandDims", [("x", f32(3, 4)), ("axis", None)], ["y"], inits={"axis": np.array(1, np.int32)}),
        Case("ms_trilu", "Trilu", [("x", f32(4, 5))], ["y"], {"upper": 0}),
        Case("ms_range", "Range", [("start", None), ("limit", None), ("delta", None)], ["y"],
             inits={"start": np.array(1.0, np.float32), "limit": np.array(7.0, np.float32), "delta": np.array(1.5, np.float32)}),
        Case("ms_gather_nd", "GatherND", [("x", f32(3, 4, 5)), ("i", None)], ["y"], inits={"i": np.array([[0, 1], [2, 3]], np.int64)}),
        Case("ms_pad", "Pad", [("x", f32(2, 3)), ("pads", None)], ["y"], {"mode": "constant"}, inits={"pads": np.array([0, 1, 1, 0], np.int64)}),
        Case("ms_crop_and_resize", "CropAndResize",
             [("x", f32(2, 3, 8, 8)), ("rois", np.array([[0.1, 0.1, 0.8, 0.9], [0.0, 0.2, 0.5, 0.6]], np.float32)),
              ("bi", np.array([0, 1], np.int32)), ("size", None)], ["y"], {"mode": "bilinear"}, inits={"size": np.array([4, 5], np.int32)}),
        Case("ms_murmur_hash3", "MurmurHash3", [("x", RNG.integers(-1000, 1000, (6,), dtype=np.int32))], ["y"], {"seed": 42, "positive": 0}),
        Case("ms_word_conv_embedding", "WordConvEmbedding", [("seq", RNG.integers(0, 10, (3, 6), dtype=np.int32)), ("w", None), ("b", None), ("c", None)],
             ["y"], {"conv_window_size": 2, "embedding_size": 4, "char_embedding_size": 3},
             inits={"w": f32(4, 1, 2, 3), "b": f32(4), "c": f32(10, 3)}),
        Case("ms_ngram_repeat_block", "NGramRepeatBlock", [("ids", np.array([[1, 2, 3, 1, 2], [4, 4, 4, 5, 6]], np.int64)), ("scores", f32(2, 8))],
             ["y"], {"ngram_size": 3}),
        Case("ms_nhwc_max_pool", "NhwcMaxPool", [("x", u8(1, 6, 6, 3))], ["y"], {"kernel_shape": [2, 2], "strides": [2, 2]}),
        Case("ms_rfft", "Rfft", [("x", f32(2, 8))], ["y"], {"signal_ndim": 1}),
        Case("ms_complex_mul", "ComplexMul", [("a", f32(3, 2)), ("b", f32(3, 2))], ["y"], ref=lambda v: [ref_complex_mul(v["a"], v["b"], False)]),
        Case("ms_complex_mul_conj", "ComplexMulConj", [("a", f32(2, 3, 2)), ("b", f32(3, 2))], ["y"], ref=lambda v: [ref_complex_mul(v["a"], v["b"], True)]),
        Case("ms_unfold_tensor", "UnfoldTensor", [("x", f32(2, 7))], ["y"], {"dim": 1, "size": 3, "step": 2}),
        Case("ms_torch_embedding", "TorchEmbedding", [("w", f32(10, 4)), ("i", np.array([[1, 3], [9, 0]], np.int64))], ["y"]),
        Case("ms_relative_position_bias", "RelativePositionBias", [("t", None), ("ql", None), ("kl", None)], ["y"], {"max_distance": 8},
             inits={"t": f32(8, N), "ql": np.array(S, np.int64), "kl": np.array(S, np.int64)}),
        Case("ms_moe_relu", "MoE", [("x", f32(4, 8)), ("router", f32(4, 3)), ("w1", None), ("b1", None), ("w2", None), ("b2", None)], ["y"],
             {"k": 2, "activation_type": "relu"},
             inits={"w1": f32(3, 16, 8, scale=0.3), "b1": f32(3, 16), "w2": f32(3, 8, 16, scale=0.3), "b2": f32(3, 8)}),
        Case("ms_moe_silu_normalized", "MoE",
             [("x", f32(4, 8)), ("router", f32(4, 3)), ("w1", None), ("b1", None), ("w2", None), ("b2", None)], ["y"],
             {"k": 2, "activation_type": "silu", "normalize_routing_weights": 1},
             inits={"w1": f32(3, 16, 8, scale=0.3), "b1": f32(3, 16), "w2": f32(3, 8, 16, scale=0.3), "b2": f32(3, 8)}),
        Case("ms_moe_swiglu", "MoE",
             [("x", f32(4, 8)), ("router", f32(4, 3)), ("w1", None), ("b1", None), ("w2", None), ("b2", None)], ["y"],
             {"k": 2, "activation_type": "swiglu", "swiglu_fusion": 1, "activation_alpha": 1.702, "activation_beta": 1.0, "swiglu_limit": 7.0},
             inits={"w1": f32(3, 32, 8, scale=0.3), "b1": f32(3, 32), "w2": f32(3, 8, 16, scale=0.3), "b2": f32(3, 8)}),
    ]
    return out


def write(case, root):
    m = case.model()
    try:
        session = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        results = session.run(None, case.feeds())
    except Exception as e:  # noqa: BLE001 - every failure means "no CPU reference"
        if case.ref is None:
            return f"skipped ({str(e).splitlines()[0][:160]})"
        results = [np.asarray(r, np.float32) for r in case.ref({**case.inits, **case.feeds()})]
    fix_output_types(m, results)
    directory = os.path.join(root, case.name)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(os.path.join(directory, "test_data_set_0"))
    with open(os.path.join(directory, "model.onnx"), "wb") as f:
        f.write(m.SerializeToString())
    for j, (name, value) in enumerate(case.feeds().items()):
        with open(os.path.join(directory, "test_data_set_0", f"input_{j}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(value, name).SerializeToString())
    for j, (vi, value) in enumerate(zip(m.graph.output, results)):
        with open(os.path.join(directory, "test_data_set_0", f"output_{j}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(np.asarray(value), vi.name).SerializeToString())
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/models/onnx/sharplight-contrib-tests"))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    written = 0
    for case in cases():
        status = write(case, args.output)
        written += status == "ok"
        if status != "ok":
            print(f"{case.name}: {status}")
    print(f"{written} contrib tests written to {args.output} (onnxruntime {ort.__version__})")


if __name__ == "__main__":
    main()
