using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// The ai.onnx.ml domain. Preprocessing ops are compositions of elementwise ops, gathers and matmuls; decision forests and the one-vs-one
// step of SVM classifiers use the IR ops TreeEnsemble and OneVsOne. Semantics follow onnxruntime, the reference the converters target.
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_mlOps =
    [
        "Binarizer", "Scaler", "Normalizer", "ArrayFeatureExtractor", "Imputer", "FeatureVectorizer", "OneHotEncoder", "LabelEncoder",
        "LinearRegressor", "LinearClassifier", "SVMRegressor", "SVMClassifier", "TreeEnsembleRegressor", "TreeEnsembleClassifier", "TreeEnsemble",
        "CategoryMapper",
    ];

    private void TranslateMl(OnnxNode n, TranslatedTensor?[] ins)
    {
        switch (n.OpType)
        {
            case "Binarizer":
                SetDyn(n, 0, Emit("Greater", [X(ins, 0), F32Const([n.Float("threshold", 0f)], Nm(n) + ".threshold")], null, Nm(n)));
                break;
            case "Scaler":
                Scaler(n, ins);
                break;
            case "Normalizer":
                Normalizer(n, ins);
                break;
            case "ArrayFeatureExtractor":
                ArrayFeatureExtractor(n, ins);
                break;
            case "Imputer":
                Imputer(n, ins);
                break;
            case "FeatureVectorizer":
                FeatureVectorizer(n, ins);
                break;
            case "OneHotEncoder":
                OneHotEncoder(n, ins);
                break;
            case "LabelEncoder":
                LabelEncoder(n, ins);
                break;
            case "CategoryMapper":
                CategoryMapper(n, ins);
                break;
            case "LinearRegressor":
                LinearRegressor(n, ins);
                break;
            case "LinearClassifier":
                LinearClassifier(n, ins);
                break;
            case "SVMRegressor":
                SvmRegressor(n, ins);
                break;
            case "SVMClassifier":
                SvmClassifier(n, ins);
                break;
            case "TreeEnsembleRegressor" or "TreeEnsembleClassifier" or "TreeEnsemble":
                TreeEnsembleOp(n, ins);
                break;
            default:
                throw new NotSupportedException($"ai.onnx.ml.{n.OpType} is not supported");
        }
    }

    private void Scaler(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var shifted = Emit("Sub", [X(ins, 0), F32Const(n.Floats("offset") ?? [0f], name + ".offset")], null, name + ".shifted");
        SetDyn(n, 0, Emit("Mul", [shifted, F32Const(n.Floats("scale") ?? [1f], name + ".scale")], null, name));
    }

    /// <summary>Row normalization over the last axis: MAX divides by the row maximum (onnxruntime), L1 and L2 by the norm.</summary>
    private void Normalizer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        string mode = n.Str("norm", "MAX") switch
        {
            "MAX" => "max",
            "L1" => "l1",
            "L2" => "l2",
            var other => throw new NotSupportedException($"Normalizer norm '{other}'"),
        };
        var norm = Emit("Reduce", [x], Attributes.From(("mode", mode), ("axes", new[] { -1L }), ("keepdims", true)), Nm(n) + ".norm");
        SetDyn(n, 0, Emit("Div", [x, norm], null, Nm(n)));
    }

    /// <summary>Gathers the flattened indices along the last axis; a 1-D input becomes one row (onnxruntime).</summary>
    private void ArrayFeatureExtractor(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!.V is { } v && v.Type == DataType.I32 ? v : X(ins, 0);
        var idx = Idx(ins[1]!);
        idx = Rs(idx, idx.Shape.Elements());
        if (x.Rank == 1)
        {
            x = Rs(x, 1, x.Shape[0].Fixed);
        }

        SetDyn(n, 0, Emit("GatherAxis", [x, idx], Attributes.From(("axis", (long)(x.Rank - 1))), Nm(n)));
    }

    private void Imputer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        string name = Nm(n);
        bool floats = n.Floats("imputed_value_floats") is { Length: > 0 };
        float[] imputed = floats ? n.Floats("imputed_value_floats")!
            : n.Ints("imputed_value_int64s")?.Select(i => (float)i).ToArray() ?? throw new InvalidDataException("Imputer has no imputed values");
        float replaced = floats ? n.Float("replaced_value_float", 0f) : n.Int("replaced_value_int64", 0);
        var hit = float.IsNaN(replaced)
            ? Emit("IsNaN", [x], null, name + ".missing")
            : Emit("Equal", [x, F32Const([replaced], name + ".replaced")], null, name + ".missing");
        SetDyn(n, 0, Emit("Where", [hit, F32Const(imputed, name + ".imputed"), x], null, name));
    }

    /// <summary>Concatenates the inputs as [N, d_i] columns, each truncated or zero-padded to its <c>inputdimensions</c> entry.</summary>
    private void FeatureVectorizer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var dims = n.Ints("inputdimensions") ?? throw new InvalidDataException("FeatureVectorizer needs inputdimensions");
        var parts = new Value[ins.Length];
        for (int i = 0; i < ins.Length; i++)
        {
            var x = X(ins, i);
            x = x.Rank == 1 ? Rs(x, x.Shape[0].Fixed, 1) : x;
            long width = x.Shape[-1].Fixed;
            if (width > dims[i])
            {
                x = _b.Slice(x, -1, 0, dims[i]);
            }
            else if (width < dims[i])
            {
                x = Emit("Pad", [x], Attributes.From(("pads", new long[] { 0, 0, 0, dims[i] - width }), ("mode", "constant"), ("value", 0f)));
            }

            parts[i] = x;
        }

        SetDyn(n, 0, parts.Length == 1 ? parts[0] : ConcatMany(1, parts));
    }

    /// <summary>One column per category: out[..., j] = (x == cats[j]).</summary>
    private void OneHotEncoder(OnnxNode n, TranslatedTensor?[] ins)
    {
        var cats = n.Ints("cats_int64s") is { Length: > 0 } c ? c : throw new NotSupportedException("OneHotEncoder with string categories on numeric input");
        var x = X(ins, 0);
        var column = Rs(x, [.. Fixed(x), 1]);
        SetDyn(n, 0, Emit("Equal", [column, F32Const([.. cats.Select(v => (float)v)], Nm(n) + ".cats")], null, Nm(n)));
    }

    /// <summary>LabelEncoder (int64, float or string keys and values, as attributes or tensors) on the IR LabelEncode op.</summary>
    private void LabelEncoder(OnnxNode n, TranslatedTensor?[] ins)
    {
        var (keys, keyType) = MlValues(n, "keys");
        var (values, valueType) = MlValues(n, "values");
        object fallback = n.Tensor("default_tensor") is { } t ? OnnxConst.FromTensor(t) is { Strings: { } s } ? s[0] : OnnxConst.FromTensor(t).Floats()[0]
            : valueType == OnnxType.String ? n.Str("default_string", "_Unused")
            : OnnxType.IsFloat(valueType) ? (double)n.Float("default_float", -0f) : n.Int("default_int64", -1);
        SetDyn(n, 0, Lookup(ins[0]!, keyType, keys, values, valueType, fallback, Nm(n)));
    }

    /// <summary>CategoryMapper: strings to integers by <c>cats_strings</c> / <c>cats_int64s</c>, or back, with a default for unknown entries.</summary>
    private void CategoryMapper(OnnxNode n, TranslatedTensor?[] ins)
    {
        string[] strings = n.Attributes.TryGetValue("cats_strings", out var s) ? [.. s.Strings] : [];
        double[] ints = (n.Ints("cats_int64s") ?? []).Select(v => (double)v).ToArray();
        var y = IsStringTensor(ins[0])
            ? Lookup(ins[0]!, OnnxType.String, strings, ints, OnnxType.Int64, (double)n.Int("default_int64", -1), Nm(n))
            : Lookup(ins[0]!, OnnxType.Int64, ints, strings, OnnxType.String, n.Str("default_string", "_Unused"), Nm(n));
        SetDyn(n, 0, y);
    }

    /// <summary>Elementwise dictionary lookup (IR LabelEncode) of x, whose ONNX type is <paramref name="keyType"/>.</summary>
    private Value Lookup(TranslatedTensor x, int keyType, object keys, object values, int valueType, object fallback, string name)
    {
        var input = keyType == OnnxType.String ? StrValue(x, name + ".x") : OnnxType.IsFloat(keyType) ? AsValue(x) : Idx(x);
        string type = valueType == OnnxType.String ? "str" : OnnxType.IsFloat(valueType) ? "f32" : "i32";
        return Emit("LabelEncode", [input], Attributes.From(("keys", keys), ("values", values), ("default", fallback), ("type", type)), name);
    }

    /// <summary>The keys or values of a LabelEncoder (<c>keys_int64s</c>, <c>keys_floats</c>, <c>keys_strings</c>, <c>keys_tensor</c>, ...):
    /// double[] or string[], and their ONNX type.</summary>
    private static (object Values, int Type) MlValues(OnnxNode n, string what)
    {
        if (n.Tensor(what + "_tensor") is { } tensor)
        {
            var c = OnnxConst.FromTensor(tensor);
            return c.Strings is { } strings ? (strings, OnnxType.String) : (c.Floats(), c.ElemType);
        }

        if (n.Ints(what + "_int64s") is { } ints)
        {
            return (ints.Select(v => (double)v).ToArray(), OnnxType.Int64);
        }

        if (n.Floats(what + "_floats") is { } floats)
        {
            return (floats.Select(v => (double)v).ToArray(), OnnxType.Float);
        }

        return n.Attributes.TryGetValue(what + "_strings", out var s) ? (s.Strings.ToArray(), OnnxType.String)
            : throw new InvalidDataException($"LabelEncoder has no {what}");
    }

    // ---- shared helpers ------------------------------------------------------------------------------------------------

    private Value F32Const(float[] values, string name)
        => ConstValue(OnnxConst.FromFloats(OnnxType.Float, [values.Length], [.. values.Select(v => (double)v)], name), name);

    private Value I32Const(long[] values, string name) => Idx(TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [values.Length], values, name)));

    /// <summary>The ONNX ML post_transform of scores [N, C] (along the last axis).</summary>
    private Value PostTransform(Value scores, string transform, string name)
    {
        switch (transform)
        {
            case "NONE":
                return scores;
            case "LOGISTIC":
                return Emit("Sigmoid", [scores], null, name + ".logistic");
            case "SOFTMAX":
                return Emit("Softmax", [scores], null, name + ".softmax");
            case "SOFTMAX_ZERO":
                {
                    // Softmax that keeps exact zeros at zero (onnxruntime: |v| <= 1e-7 counts as zero); an all-zero row becomes 0.5.
                    var max = Emit("ReduceMax", [scores], null, name + ".max");
                    var nonZero = Emit("Greater", [Emit("Abs", [scores]), F32Const([1e-7f], name + ".eps")]);
                    var e = Emit("Where", [nonZero, Emit("Exp", [Emit("Sub", [scores, max])]), Emit("Mul", [scores, Emit("Exp", [Emit("Neg", [max])])])]);
                    var sum = Emit("ReduceSum", [e]);
                    var zeroSum = Emit("Equal", [sum, F32Const([0f], name + ".zero")]);
                    return Emit("Where", [zeroSum, F32Const([0.5f], name + ".half"), Emit("Div", [e, sum])], null, name + ".softmax_zero");
                }

            case "PROBIT":
                return Probit(scores, name);
            default:
                throw new NotSupportedException($"post_transform '{transform}'");
        }
    }

    /// <summary>√2 · erfinv(2p - 1) with Winitzki's approximation of erfinv (as onnxruntime).</summary>
    private Value Probit(Value p, string name)
    {
        var x = _b.Scale(p, 2f, -1f);
        var sign = Emit("Sign", [x]);
        var oneMinus = Emit("Mul", [_b.Scale(x, -1f, 1f), _b.Scale(x, 1f, 1f)]);
        var log = Emit("Log", [oneMinus]);
        var v = _b.Scale(log, 0.5f, 2f / (MathF.PI * 0.147f));
        var v2 = _b.Scale(log, 1f / 0.147f);
        var root = Emit("Sqrt", [Emit("Sub", [Emit("Mul", [v, v]), v2])]);
        var erfinv = Emit("Mul", [sign, Emit("Sqrt", [Emit("Sub", [root, v])])]);
        return _b.Scale(erfinv, 1.41421356f, 0f, name + ".probit");
    }

    /// <summary>Class labels (integers, or strings through a LabelEncode lookup) selected by an i32 index tensor.</summary>
    private Value ClassLabels(OnnxNode n, Value index, string name)
    {
        if ((n.Ints("classlabels_int64s") ?? n.Ints("classlabels_ints")) is { } labels)
        {
            return Emit("GatherAxis", [I32Const(labels, name + ".labels"), index], Attributes.From(("axis", 0L)), name);
        }

        string[] strings = n.Attributes.TryGetValue("classlabels_strings", out var s) ? [.. s.Strings]
            : throw new InvalidDataException($"{n.OpType} has no class labels");
        double[] positions = [.. Enumerable.Range(0, strings.Length).Select(i => (double)i)];
        return Emit("LabelEncode", [index], Attributes.From(("keys", positions), ("values", strings), ("default", ""), ("type", "str")), name);
    }

    private int ClassCount(OnnxNode n)
        => Math.Max(n.Ints("classlabels_int64s")?.Length ?? n.Ints("classlabels_ints")?.Length ?? 0, n.Attributes.TryGetValue("classlabels_strings", out var s) ? s.Strings.Count : 0);

    /// <summary>Index of the largest score per row (first on ties).</summary>
    private Value ArgMaxRows(Value scores, string name)
        => Emit("ArgReduce", [scores], Attributes.From(("mode", "max"), ("axis", -1L), ("keepdims", false)), name + ".argmax");
}
