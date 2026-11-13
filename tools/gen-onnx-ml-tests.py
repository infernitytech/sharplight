#!/usr/bin/env python3
"""Writes extra conformance tests for the ai.onnx.ml domain, random-length signal transforms and string ops, with onnxruntime outputs.

The ONNX backend node tests cover only a handful of ai.onnx.ml operators. This script builds small models in the layouts the converters
(skl2onnx, onnxmltools) emit and stores them in the node-test format (model.onnx + test_data_set_0/input_k.pb, output_k.pb), with the
expected outputs computed by onnxruntime's CPU provider, the de facto reference for these operators.

Usage (in a venv with onnx, onnxruntime and numpy, never the system Python):
    python -I tools/gen-onnx-ml-tests.py [-o ~/models/onnx/sharplight-ml-tests]

tests/Sharplight.Onnx.Conformance (ExtraConformanceTests) reads the directory from SHARPLIGHT_ONNX_EXTRA_TESTS, else
~/models/onnx/sharplight-ml-tests.
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

ML = "ai.onnx.ml"
FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64
STRING = TensorProto.STRING


def model(name, nodes, inputs, outputs, ml_opset=3, opset=21, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=list(initializers))
    opsets = [helper.make_opsetid("", opset), helper.make_opsetid(ML, ml_opset)]
    m = helper.make_model(graph, opset_imports=opsets, ir_version=10)
    onnx.checker.check_model(m)
    return m


def vi(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def ml(op, ins, outs, **attrs):
    return helper.make_node(op, ins, outs, domain=ML, **attrs)


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


def simple_ops(root, rng):
    x = rng.standard_normal((5, 4)).astype(np.float32)
    feeds = {"X": x}
    xin = [vi("X", FLOAT, [5, 4])]
    write(root, "ml_binarizer", model("b", [ml("Binarizer", ["X"], ["Y"], threshold=0.3)], xin, [vi("Y", FLOAT, [5, 4])]), feeds)
    write(root, "ml_scaler", model("s", [ml("Scaler", ["X"], ["Y"], offset=[0.1, -0.2, 0.3, 0.0], scale=[2.0, 0.5, -1.0, 3.0])], xin,
                                   [vi("Y", FLOAT, [5, 4])]), feeds)
    write(root, "ml_scaler_broadcast", model("s", [ml("Scaler", ["X"], ["Y"], offset=[0.25], scale=[4.0])], xin, [vi("Y", FLOAT, [5, 4])]), feeds)
    for norm in ["MAX", "L1", "L2"]:
        write(root, f"ml_normalizer_{norm.lower()}", model("n", [ml("Normalizer", ["X"], ["Y"], norm=norm)], xin, [vi("Y", FLOAT, [5, 4])]), feeds)
    idx = np.array([[3, 0, 3]], dtype=np.int64)
    write(root, "ml_array_feature_extractor", model("a", [ml("ArrayFeatureExtractor", ["X", "I"], ["Y"])],
                                                    [vi("X", FLOAT, [5, 4])], [vi("Y", FLOAT, [5, 3])], initializers=[numpy_helper.from_array(idx, "I")]),
          feeds)
    xn = x.copy()
    xn[1, 2] = np.nan
    xn[3, 0] = np.nan
    write(root, "ml_imputer_nan", model("i", [ml("Imputer", ["X"], ["Y"], imputed_value_floats=[1.0, 2.0, 3.0, 4.0], replaced_value_float=np.nan)],
                                        xin, [vi("Y", FLOAT, [5, 4])]), {"X": xn})
    xz = x.copy()
    xz[2, 1] = 0.0
    write(root, "ml_imputer_value", model("i", [ml("Imputer", ["X"], ["Y"], imputed_value_floats=[-9.0], replaced_value_float=0.0)],
                                          xin, [vi("Y", FLOAT, [5, 4])]), {"X": xz})
    b = rng.standard_normal((5, 4)).astype(np.float32)
    write(root, "ml_feature_vectorizer", model("f", [ml("FeatureVectorizer", ["X", "B"], ["Y"], inputdimensions=[4, 2])],
                                               [vi("X", FLOAT, [5, 4]), vi("B", FLOAT, [5, 4])], [vi("Y", FLOAT, [5, 6])]), {"X": x, "B": b})
    ints = np.array([[1, 5, 7], [7, 1, 2], [5, 5, 9]], dtype=np.int64)
    write(root, "ml_one_hot_encoder", model("o", [ml("OneHotEncoder", ["X"], ["Y"], cats_int64s=[1, 5, 7, 9], zeros=1)],
                                            [vi("X", INT64, [3, 3])], [vi("Y", FLOAT, [3, 3, 4])]), {"X": ints})
    write(root, "ml_label_encoder_int_int", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_int64s=[1, 5, 7], values_int64s=[10, 50, 70],
                                                           default_int64=-1)], [vi("X", INT64, [3, 3])], [vi("Y", INT64, [3, 3])], ml_opset=2), {"X": ints})
    write(root, "ml_label_encoder_int_float", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_int64s=[1, 5, 7], values_floats=[0.5, 1.5, 2.5],
                                                             default_float=-0.25)], [vi("X", INT64, [3, 3])], [vi("Y", FLOAT, [3, 3])], ml_opset=2),
          {"X": ints})
    floats = np.array([[0.5, 1.5, 2.0], [2.5, 0.5, 0.5]], dtype=np.float32)
    write(root, "ml_label_encoder_float_int", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_floats=[0.5, 1.5, 2.5], values_int64s=[1, 2, 3],
                                                             default_int64=0)], [vi("X", FLOAT, [2, 3])], [vi("Y", INT64, [2, 3])], ml_opset=2),
          {"X": floats})
    keys = numpy_helper.from_array(np.array([1, 5, 7], dtype=np.int64), "keys_tensor")
    values = numpy_helper.from_array(np.array([0.5, 1.5, 2.5], dtype=np.float32), "values_tensor")
    default = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "default_tensor")
    write(root, "ml_label_encoder_tensor_attrs", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_tensor=keys, values_tensor=values,
                                                                default_tensor=default)], [vi("X", INT64, [3, 3])], [vi("Y", FLOAT, [3, 3])],
                                                         ml_opset=4), {"X": ints})


def linear_ops(root, rng):
    x = rng.standard_normal((6, 4)).astype(np.float32)
    xin = [vi("X", FLOAT, [6, 4])]
    coef = rng.standard_normal(8).astype(np.float32).tolist()
    write(root, "ml_linear_regressor", model("lr", [ml("LinearRegressor", ["X"], ["Y"], coefficients=coef, intercepts=[0.5, -1.0], targets=2)],
                                             xin, [vi("Y", FLOAT, [6, 2])]), {"X": x})
    coef3 = rng.standard_normal(12).astype(np.float32).tolist()
    for post in ["NONE", "SOFTMAX", "LOGISTIC", "SOFTMAX_ZERO"]:
        node = ml("LinearClassifier", ["X"], ["L", "S"], coefficients=coef3, intercepts=[0.1, 0.2, -0.3], classlabels_ints=[2, 4, 6],
                  post_transform=post)
        write(root, f"ml_linear_classifier_{post.lower()}", model("lc", [node], xin, [vi("L", INT64, [6]), vi("S", FLOAT, [6, 3])]), {"X": x})
    for post in ["NONE", "LOGISTIC"]:
        node = ml("LinearClassifier", ["X"], ["L", "S"], coefficients=coef3[:4], intercepts=[0.1], classlabels_ints=[0, 1], post_transform=post)
        write(root, f"ml_linear_classifier_binary_{post.lower()}", model("lc", [node], xin, [vi("L", INT64, [6]), vi("S", FLOAT, [6, 2])]), {"X": x})
    node = ml("LinearClassifier", ["X"], ["L", "S"], coefficients=coef3, intercepts=[0.1, 0.2, -0.3], classlabels_strings=["a", "b", "c"],
              post_transform="SOFTMAX")
    write(root, "ml_linear_classifier_strings", model("lc", [node], xin, [vi("L", STRING, [6]), vi("S", FLOAT, [6, 3])]), {"X": x})


def svm_ops(root, rng):
    x = rng.standard_normal((6, 3)).astype(np.float32)
    xin = [vi("X", FLOAT, [6, 3])]
    sv = rng.standard_normal((4, 3)).astype(np.float32)
    for kernel, params in [("LINEAR", [0.0, 0.0, 0.0]), ("RBF", [0.4, 0.0, 0.0]), ("POLY", [0.3, 0.5, 3.0]), ("SIGMOID", [0.2, 0.1, 0.0])]:
        node = ml("SVMRegressor", ["X"], ["Y"], coefficients=[0.5, -0.25, 1.0, 0.75], kernel_params=params, kernel_type=kernel, n_supports=4,
                  rho=[0.3], support_vectors=sv.ravel().tolist())
        write(root, f"ml_svm_regressor_{kernel.lower()}", model("svr", [node], xin, [vi("Y", FLOAT, [6, 1])]), {"X": x})
    node = ml("SVMRegressor", ["X"], ["Y"], coefficients=[1.0, 2.0, -1.0], kernel_params=[0.0, 0.0, 0.0], kernel_type="LINEAR", n_supports=0, rho=[0.5])
    write(root, "ml_svm_regressor_linear_coefficients", model("svr", [node], xin, [vi("Y", FLOAT, [6, 1])]), {"X": x})
    # Three classes one-vs-one (libsvm layout): 2 + 1 + 2 support vectors, 3 pairwise decision functions.
    sv5 = rng.standard_normal((5, 3)).astype(np.float32)
    coefs = rng.standard_normal(2 * 5).astype(np.float32).tolist()
    for kernel in ["RBF", "LINEAR"]:
        common = dict(classlabels_ints=[0, 1, 2], coefficients=coefs, kernel_params=[0.3, 0.0, 3.0], kernel_type=kernel,
                      rho=[0.1, -0.2, 0.3], support_vectors=sv5.ravel().tolist(), vectors_per_class=[2, 1, 2])
        write(root, f"ml_svm_classifier_ovo_{kernel.lower()}", model("svc", [ml("SVMClassifier", ["X"], ["L", "S"], **common)], xin,
                                                                       [vi("L", INT64, [6]), vi("S", FLOAT, [6, 3])]), {"X": x})
        write(root, f"ml_svm_classifier_ovo_prob_{kernel.lower()}",
              model("svc", [ml("SVMClassifier", ["X"], ["L", "S"], prob_a=[-1.5, -2.0, -0.7], prob_b=[0.1, -0.2, 0.05], **common)], xin,
                    [vi("L", INT64, [6]), vi("S", FLOAT, [6, 3])]), {"X": x})
    binary = dict(classlabels_ints=[0, 1], coefficients=coefs[:3], kernel_params=[0.3, 0.0, 3.0], kernel_type="RBF", rho=[0.2],
                  support_vectors=sv5[:3].ravel().tolist(), vectors_per_class=[2, 1])
    write(root, "ml_svm_classifier_binary", model("svc", [ml("SVMClassifier", ["X"], ["L", "S"], **binary)], xin,
                                                  [vi("L", INT64, [6]), vi("S", FLOAT, [6, 2])]), {"X": x})
    write(root, "ml_svm_classifier_binary_prob", model("svc", [ml("SVMClassifier", ["X"], ["L", "S"], prob_a=[-1.2], prob_b=[0.3], **binary)], xin,
                                                       [vi("L", INT64, [6]), vi("S", FLOAT, [6, 2])]), {"X": x})
    linear = dict(classlabels_ints=[3, 5, 7], coefficients=rng.standard_normal(9).astype(np.float32).tolist(), kernel_params=[0.0, 0.0, 0.0], kernel_type="LINEAR",
                  rho=[0.1, 0.2, 0.3])
    write(root, "ml_svm_classifier_linear_ovr", model("svc", [ml("SVMClassifier", ["X"], ["L", "S"], **linear)], xin,
                                                      [vi("L", INT64, [6]), vi("S", FLOAT, [6, 3])]), {"X": x})


def tree_attributes(rng, n_targets, leaf_key):
    """Two trees over 3 features exercising every branch mode; leaves carry weights for n_targets targets."""
    modes = ["BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "LEAF", "LEAF", "LEAF", "LEAF",
             "BRANCH_GT", "BRANCH_EQ", "BRANCH_NEQ", "LEAF", "LEAF", "LEAF", "LEAF"]
    a = {
        "nodes_treeids": [0] * 7 + [1] * 7,
        "nodes_nodeids": list(range(7)) * 2,
        "nodes_featureids": [0, 1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0],
        "nodes_values": [0.1, -0.3, 0.5, 0, 0, 0, 0, 0.2, 0.0, 0.7, 0, 0, 0, 0],
        "nodes_modes": modes,
        "nodes_truenodeids": [1, 3, 5, 0, 0, 0, 0, 1, 3, 5, 0, 0, 0, 0],
        "nodes_falsenodeids": [2, 4, 6, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0],
        "nodes_missing_value_tracks_true": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    }
    tree, node, target, weight = [], [], [], []
    for t in range(2):
        for nid in range(3, 7):
            for k in range(n_targets):
                if (nid + k + t) % 3 == 0 and n_targets > 1:
                    continue
                tree.append(t)
                node.append(nid)
                target.append(k)
                weight.append(float(np.round(rng.standard_normal(), 3)))
    a[f"{leaf_key}_treeids"] = tree
    a[f"{leaf_key}_nodeids"] = node
    a[f"{leaf_key}_ids"] = target
    a[f"{leaf_key}_weights"] = weight
    return a


def tree_ops(root, rng):
    x = rng.standard_normal((8, 3)).astype(np.float32)
    x[1, 0] = np.nan
    x[2, 1] = 0.0
    x[3, 0] = 0.7
    xin = [vi("X", FLOAT, [8, 3])]
    for agg in ["SUM", "AVERAGE", "MIN", "MAX"]:
        a = tree_attributes(rng, 2, "target")
        node = ml("TreeEnsembleRegressor", ["X"], ["Y"], n_targets=2, aggregate_function=agg, base_values=[0.5, -0.5], **a)
        write(root, f"ml_tree_regressor_{agg.lower()}", model("tr", [node], xin, [vi("Y", FLOAT, [8, 2])]), {"X": x})
    for post in ["NONE", "SOFTMAX", "LOGISTIC", "SOFTMAX_ZERO"]:
        a = tree_attributes(rng, 3, "class")
        node = ml("TreeEnsembleClassifier", ["X"], ["L", "S"], classlabels_int64s=[1, 2, 3], post_transform=post, **a)
        write(root, f"ml_tree_classifier_{post.lower()}", model("tc", [node], xin, [vi("L", INT64, [8]), vi("S", FLOAT, [8, 3])]), {"X": x})
    a = tree_attributes(rng, 3, "class")
    node = ml("TreeEnsembleClassifier", ["X"], ["L", "S"], classlabels_int64s=[1, 2, 3], base_values=[0.1, -0.2, 0.3], **a)
    write(root, "ml_tree_classifier_base", model("tc", [node], xin, [vi("L", INT64, [8]), vi("S", FLOAT, [8, 3])]), {"X": x})
    for post in ["NONE", "LOGISTIC"]:
        a = tree_attributes(rng, 1, "class")
        node = ml("TreeEnsembleClassifier", ["X"], ["L", "S"], classlabels_int64s=[0, 1], post_transform=post, **a)
        write(root, f"ml_tree_classifier_binary_{post.lower()}", model("tc", [node], xin, [vi("L", INT64, [8]), vi("S", FLOAT, [8, 2])]), {"X": x})
        node = ml("TreeEnsembleClassifier", ["X"], ["L", "S"], classlabels_int64s=[0, 1], post_transform=post, base_values=[0.2, -0.1], **a)
        write(root, f"ml_tree_classifier_binary_base_{post.lower()}", model("tc", [node], xin, [vi("L", INT64, [8]), vi("S", FLOAT, [8, 2])]),
              {"X": x})
    a = tree_attributes(rng, 3, "class")
    node = ml("TreeEnsembleClassifier", ["X"], ["L", "S"], classlabels_strings=["x", "y", "z"], post_transform="SOFTMAX", **a)
    write(root, "ml_tree_classifier_strings", model("tc", [node], xin, [vi("L", STRING, [8]), vi("S", FLOAT, [8, 3])]), {"X": x})

    # TreeEnsemble (ai.onnx.ml 5): set membership, every aggregate function.
    splits = numpy_helper.from_array(np.array([0.1, np.nan, -0.2], dtype=np.float32), "s")
    for agg in [0, 1, 2, 3]:
        node = ml("TreeEnsemble", ["X"], ["Y"], n_targets=2, aggregate_function=agg, post_transform=0,
                  nodes_featureids=[0, 1, 2], nodes_modes=numpy_helper.from_array(np.array([0, 6, 3], dtype=np.uint8), "m"),
                  nodes_splits=splits, nodes_truenodeids=[1, 0, 0], nodes_trueleafs=[0, 1, 1], nodes_falsenodeids=[2, 1, 2],
                  nodes_falseleafs=[1, 1, 1], nodes_missing_value_tracks_true=[0, 1, 0],
                  membership_values=numpy_helper.from_array(np.array([0.0, 1.5, np.nan], dtype=np.float32), "mv"),
                  leaf_targetids=[0, 1, 0], leaf_weights=numpy_helper.from_array(np.array([1.0, 2.5, -1.0], dtype=np.float32), "lw"),
                  tree_roots=[0, 2])
        xm = x.copy()
        xm[0, 1] = 1.5
        xm[4, 1] = 0.0
        write(root, f"ml_tree_ensemble_v5_agg{agg}", model("te", [node], xin, [vi("Y", FLOAT, [8, 2])], ml_opset=5), {"X": xm})


def signal_ops(root, rng):
    """DFT/STFT lengths beyond the dense-basis limit (FFT path); references from numpy's FFT (onnxruntime accumulates in float)."""
    x = rng.standard_normal((2, 1500, 1)).astype(np.float32)
    length = numpy_helper.from_array(np.array(1500, dtype=np.int64), "len")
    axis = numpy_helper.from_array(np.array(1, dtype=np.int64), "axis")
    node = helper.make_node("DFT", ["X", "len", "axis"], ["Y"], onesided=1)
    write(root, "signal_dft_long_rfft", model("d", [node], [vi("X", FLOAT, [2, 1500, 1])], [vi("Y", FLOAT, [2, 751, 2])],
                                              initializers=[length, axis]), {"X": x}, reference=True)
    xc = rng.standard_normal((2, 2048, 2)).astype(np.float32)
    length2 = numpy_helper.from_array(np.array(2048, dtype=np.int64), "len")
    node = helper.make_node("DFT", ["X", "len", "axis"], ["Y"], inverse=1)
    write(root, "signal_dft_long_inverse", model("d", [node], [vi("X", FLOAT, [2, 2048, 2])], [vi("Y", FLOAT, [2, 2048, 2])],
                                                 initializers=[length2, axis]), {"X": xc}, reference=True)
    signal = rng.standard_normal((1, 8000, 1)).astype(np.float32)
    step = numpy_helper.from_array(np.array(256, dtype=np.int64), "step")
    window = numpy_helper.from_array(np.hanning(1024).astype(np.float32), "window")
    node = helper.make_node("STFT", ["X", "step", "window"], ["Y"])
    write(root, "signal_stft_long_window", model("s", [node], [vi("X", FLOAT, [1, 8000, 1])], [vi("Y", FLOAT, [1, 28, 513, 2])],
                                                 initializers=[step, window]), {"X": signal}, reference=True)


def string_ops(root, rng):
    """String tensors: lookups (LabelEncoder, CategoryMapper), TfIdfVectorizer over strings, casts and string constants."""
    words = np.array([["red", "green", "blue"], ["blue", "teal", "red"]], dtype=object)
    sin = [vi("X", STRING, [2, 3])]
    feeds = {"X": words}
    write(root, "ml_label_encoder_string_float", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_strings=["red", "green", "blue"],
                                                                values_floats=[0.5, 1.5, 2.5], default_float=-1.0)], sin,
                                                         [vi("Y", FLOAT, [2, 3])], ml_opset=2), feeds)
    write(root, "ml_label_encoder_string_string", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_strings=["red", "blue"],
                                                                 values_strings=["rot", "blau"], default_string="?")], sin,
                                                          [vi("Y", STRING, [2, 3])], ml_opset=2), feeds)
    ints = np.array([[1, 2, 7], [3, 1, 1]], dtype=np.int64)
    write(root, "ml_label_encoder_int_string", model("l", [ml("LabelEncoder", ["X"], ["Y"], keys_int64s=[1, 2, 3], values_strings=["a", "b", "c"],
                                                              default_string="none")], [vi("X", INT64, [2, 3])], [vi("Y", STRING, [2, 3])],
                                                       ml_opset=2), {"X": ints})
    write(root, "ml_category_mapper_to_int", model("c", [ml("CategoryMapper", ["X"], ["Y"], cats_strings=["red", "green", "blue"],
                                                            cats_int64s=[10, 20, 30], default_int64=-7)], sin, [vi("Y", INT64, [2, 3])]), feeds)
    write(root, "ml_category_mapper_to_string", model("c", [ml("CategoryMapper", ["X"], ["Y"], cats_strings=["red", "green", "blue"],
                                                               cats_int64s=[1, 2, 3], default_string="other")], [vi("X", INT64, [2, 3])],
                                                      [vi("Y", STRING, [2, 3])]), {"X": ints})
    node = helper.make_node("TfIdfVectorizer", ["X"], ["Y"], mode="TF", min_gram_length=1, max_gram_length=2, max_skip_count=0,
                            ngram_counts=[0, 3], ngram_indexes=[0, 1, 2, 3, 4], pool_strings=["red", "blue", "teal", "blue", "teal", "red", "green"])
    write(root, "text_tfidf_strings", model("t", [node], sin, [vi("Y", FLOAT, [2, 5])]), feeds)
    numbers = np.array(["1.5", "-2", "1e3", "INF", "-0.25", "7"], dtype=object)
    write(root, "string_cast_to_float", model("c", [helper.make_node("Cast", ["X"], ["Y"], to=FLOAT)], [vi("X", STRING, [6])], [vi("Y", FLOAT, [6])]),
          {"X": numbers})
    whole = np.array(["12", "-3", "0", "40000"], dtype=object)
    write(root, "string_cast_to_int", model("c", [helper.make_node("Cast", ["X"], ["Y"], to=INT64)], [vi("X", STRING, [4])], [vi("Y", INT64, [4])]),
          {"X": whole})
    counts = np.array([3, -1, 250], dtype=np.int64)
    write(root, "string_cast_from_int", model("c", [helper.make_node("Cast", ["X"], ["Y"], to=STRING)], [vi("X", INT64, [3])],
                                              [vi("Y", STRING, [3])]), {"X": counts})
    suffix = numpy_helper.from_array(np.array(["!"], dtype=object), "suffix")
    nodes = [helper.make_node("StringConcat", ["X", "suffix"], ["C"]), helper.make_node("Equal", ["C", "X"], ["E"]),
             helper.make_node("RegexFullMatch", ["C"], ["M"], pattern="(red|blue)!")]
    write(root, "string_concat_constant", model("s", nodes, sin, [vi("C", STRING, [2, 3]), vi("E", TensorProto.BOOL, [2, 3]),
                                                                  vi("M", TensorProto.BOOL, [2, 3])], initializers=[suffix]), feeds, reference=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/models/onnx/sharplight-ml-tests"),
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()
    rng = np.random.default_rng(1234)
    os.makedirs(args.output, exist_ok=True)
    simple_ops(args.output, rng)
    linear_ops(args.output, rng)
    svm_ops(args.output, rng)
    tree_ops(args.output, rng)
    signal_ops(args.output, rng)
    string_ops(args.output, rng)
    print(f"wrote {len(os.listdir(args.output))} tests to {args.output}")


if __name__ == "__main__":
    main()
