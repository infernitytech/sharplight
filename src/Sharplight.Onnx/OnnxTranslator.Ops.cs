using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

internal sealed partial class OnnxTranslator
{
    static readonly Dictionary<string, string> s_unary = new()
    {
        ["Relu"] = "Relu", ["Sigmoid"] = "Sigmoid", ["Tanh"] = "Tanh", ["Exp"] = "Exp", ["Log"] = "Log", ["Sqrt"] = "Sqrt", ["Neg"] = "Neg",
        ["Abs"] = "Abs", ["Erf"] = "Erf", ["Reciprocal"] = "Recip", ["HardSwish"] = "HardSwish", ["Softplus"] = "Softplus", ["Mish"] = "Mish",
        ["Sin"] = "Sin", ["Cos"] = "Cos", ["Tan"] = "Tan", ["Asin"] = "Asin", ["Acos"] = "Acos", ["Atan"] = "Atan", ["Sinh"] = "Sinh", ["Cosh"] = "Cosh",
        ["Asinh"] = "Asinh", ["Acosh"] = "Acosh", ["Atanh"] = "Atanh", ["Floor"] = "Floor", ["Ceil"] = "Ceil", ["Round"] = "Round", ["Sign"] = "Sign",
        ["Softsign"] = "Softsign", ["Not"] = "Not", ["IsNaN"] = "IsNaN", ["IsInf"] = "IsInf", ["Silu"] = "Silu", ["Swish"] = "Silu",
    };

    /// <summary>ONNX unary ops with float attributes: (attribute, default) pairs, same names on the IR op.</summary>
    static readonly Dictionary<string, (string Ir, (string Name, float Def)[] Attrs)> s_unaryAttr = new()
    {
        ["Elu"] = ("Elu", [("alpha", 1f)]),
        ["Celu"] = ("Celu", [("alpha", 1f)]),
        ["Selu"] = ("Selu", [("alpha", 1.67326319217681884765625f), ("gamma", 1.05070102214813232421875f)]),
        ["LeakyRelu"] = ("LeakyRelu", [("alpha", 0.01f)]),
        ["ThresholdedRelu"] = ("ThresholdedRelu", [("alpha", 1f)]),
        ["HardSigmoid"] = ("HardSigmoid", [("alpha", 0.2f), ("beta", 0.5f)]),
    };

    static readonly Dictionary<string, string> s_binary = new()
    {
        ["Add"] = "Add", ["Sub"] = "Sub", ["Mul"] = "Mul", ["Div"] = "Div", ["Pow"] = "Pow", ["PRelu"] = "PRelu", ["Equal"] = "Equal",
        ["Less"] = "Less", ["Greater"] = "Greater", ["LessOrEqual"] = "LessEqual", ["GreaterOrEqual"] = "GreaterEqual", ["And"] = "And", ["Or"] = "Or", ["Xor"] = "Xor",
    };

    Value X(Tn?[] ins, int i, string? hint = null)
        => i < ins.Length && ins[i] is { } t ? AsValue(t, hint) : throw new InvalidDataException($"missing input {i}");

    OnnxConst ConstIn(Tn?[] ins, int i, string what)
        => i < ins.Length && ins[i] is { } t
            ? t.C ?? throw new NotSupportedException($"{what} must be a compile-time constant (initializer or constant-folded value)")
            : throw new InvalidDataException($"missing input {i} ({what})");

    OnnxConst? OptConst(Tn?[] ins, int i, string what)
        => i < ins.Length && ins[i] is { } t ? t.C ?? throw new NotSupportedException($"{what} must be a compile-time constant") : null;

    static long[] L(params long[] v) => v;

    Value Emit(string op, Value[] inputs, Attributes? attrs = null, string? name = null)
    {
        // A leading 0 in a reshape target means "keep the symbolic batch dim"; for a fixed outer dim spell it out.
        if (op == "Reshape" && attrs is not null && !Sym(inputs[0]) && inputs[0].Rank > 0 && attrs.Get<long[]>("shape") is { Length: > 0 } sh && sh[0] == 0)
        {
            sh = (long[])sh.Clone(); sh[0] = inputs[0].Shape[0].Fixed;
            attrs = Attributes.From(("shape", sh));
        }
        return _b.Emit1(op, inputs, attrs, name);
    }

    Value Rs(Value x, params long[] shape) => Emit("Reshape", [x], Attributes.From(("shape", shape)));

    void Dynamic(OnnxNode n, Tn?[] ins)
    {
        string op = n.OpType;
        if (s_unary.TryGetValue(op, out var irUnary)) { SetDyn(n, 0, Emit(irUnary, [X(ins, 0)], null, Nm(n))); return; }
        if (s_unaryAttr.TryGetValue(op, out var ua))
        {
            SetDyn(n, 0, Emit(ua.Ir, [X(ins, 0)], Attributes.From(ua.Attrs.Select(a => (a.Name, (object)n.Float(a.Name, a.Def))).ToArray()), Nm(n)));
            return;
        }
        if (s_binary.TryGetValue(op, out var irBin)) { SetDyn(n, 0, Emit(irBin, [X(ins, 0, Nm(n) + ".a"), X(ins, 1, Nm(n) + ".b")], null, Nm(n))); return; }
        if (op == "Mod") { SetDyn(n, 0, Emit("Mod", [X(ins, 0), X(ins, 1)], Attributes.From(("fmod", n.Int("fmod", 0) != 0)), Nm(n))); return; }
        switch (op)
        {
            case "Conv": Conv(n, ins); break;
            case "BatchNormalization": BatchNorm(n, ins); break;
            case "MaxPool": case "AveragePool": Pool(n, ins); break;
            case "GlobalAveragePool": case "GlobalMaxPool":
            {
                var x = X(ins, 0);
                var axes = Enumerable.Range(2, Math.Max(0, x.Rank - 2)).Select(i => (long)i).ToArray();
                SetDyn(n, 0, Emit("Reduce", [x], Attributes.From(("mode", op == "GlobalAveragePool" ? "mean" : "max"), ("axes", axes), ("keepdims", true)), Nm(n)));
                break;
            }
            case "ReduceMean": case "ReduceSum": case "ReduceMax": case "ReduceMin": case "ReduceL2": case "ReduceL1": case "ReduceProd": case "ReduceSumSquare": case "ReduceLogSum": case "ReduceLogSumExp": Reduce(n, ins); break;
            case "Gemm": Gemm(n, ins); break;
            case "MatMul": MatMul(n, ins); break;
            case "Clip":
            {
                var x = X(ins, 0);
                var items = new List<(string, object)>();
                float? lo = n.Has("min") ? n.Float("min", 0) : OptConst(ins, 1, "Clip min") is { } l ? l.ScalarF32() : null;
                float? hi = n.Has("max") ? n.Float("max", 0) : OptConst(ins, 2, "Clip max") is { } h ? h.ScalarF32() : null;
                if (lo is { } a) items.Add(("min", a));
                if (hi is { } b) items.Add(("max", b));
                SetDyn(n, 0, Emit("Clip", [x], Attributes.From(items.ToArray()), Nm(n)));
                break;
            }
            case "Gelu":
                SetDyn(n, 0, Emit(n.Str("approximate", "none") == "tanh" ? "GeluTanh" : "Gelu", [X(ins, 0)], null, Nm(n))); break;
            case "Max": case "Min": case "Sum": case "Mean":
            {
                string irOp = op == "Max" ? "Max" : op == "Min" ? "Min" : "Add";
                var acc = X(ins, 0);
                for (int i = 1; i < ins.Length; i++) acc = Emit(irOp, [acc, X(ins, i)], null, i == ins.Length - 1 && op != "Mean" ? Nm(n) : null);
                if (op == "Mean") acc = Emit("Scale", [acc], Attributes.From(("scale", 1f / ins.Length), ("bias", 0f)), Nm(n));
                SetDyn(n, 0, acc);
                break;
            }
            case "Identity": _env[n.Outputs[0]] = ins[0]!; break;
            case "Dropout":
                CheckUnusedOutputs(n, 1);
                _env[n.Outputs[0]] = ins[0]!; break;
            case "Cast": CastDyn(n, ins); break;
            case "Shape":
            {
                var d = ins[0]!.V!.Shape;
                var dims = d.Dims.Select(x => x.IsSymbolic ? OnnxConst.SymBatch : x.Fixed).ToArray();
                long s = n.Int("start", 0), e = n.Int("end", dims.Length);
                if (s < 0) s += dims.Length; if (e < 0) e += dims.Length;
                s = Math.Clamp(s, 0, dims.Length); e = Math.Clamp(e, s, dims.Length);
                _env[n.Outputs[0]] = Tn.Const(OnnxConst.FromInts(OnnxType.Int64, [e - s], dims[(int)s..(int)e]));
                break;
            }
            case "Size":
                if (Sym(ins[0]!.V!)) throw new NotSupportedException("Size of a tensor with a symbolic batch dimension");
                _env[n.Outputs[0]] = Tn.Const(OnnxConst.ScalarInt(ins[0]!.V!.Shape.Elements()));
                break;
            case "Flatten": Flatten(n, ins); break;
            case "Reshape":
                SetDyn(n, 0, ReshapeOnnx(X(ins, 0), ConstIn(ins, 1, "Reshape target shape").Ints(), n.Int("allowzero", 0) != 0, Nm(n)));
                break;
            case "Squeeze": case "Unsqueeze": SqueezeUnsqueeze(n, ins); break;
            case "Transpose":
            {
                var x = X(ins, 0);
                var perm = n.Ints("perm")?.Select(p => (int)p).ToArray() ?? Enumerable.Range(0, x.Rank).Reverse().ToArray();
                SetDyn(n, 0, Permute(x, perm));
                break;
            }
            case "Concat": Concat(n, ins); break;
            case "Slice": Slice(n, ins); break;
            case "Split": Split(n, ins); break;
            case "Gather": Gather(n, ins); break;
            case "Expand": Expand(n, ins); break;
            case "Softmax": case "LogSoftmax": Softmax(n, ins); break;
            case "LayerNormalization": LayerNorm(n, ins); break;
            case "InstanceNormalization": InstanceNorm(n, ins); break;
            case "Pad": Pad(n, ins); break;
            case "Resize": case "Upsample": Resize(n, ins); break;
            default:
                if (!More(n, ins)) throw new NotSupportedException($"{op} on a runtime-dependent tensor is not supported (only constant-foldable uses are).");
                break;
        }
    }

    // ================================ spatial ============================================================
    static long[] AutoPad(string mode, long[] inSize, long[] kernel, long[] strides, long[] dilations)
    {
        int r = inSize.Length;
        var pads = new long[2 * r];
        for (int i = 0; i < r; i++)
        {
            long outSize = (inSize[i] + strides[i] - 1) / strides[i];
            long total = Math.Max(0, (outSize - 1) * strides[i] + (kernel[i] - 1) * dilations[i] + 1 - inSize[i]);
            long small = total / 2, big = total - small;
            bool upper = mode == "SAME_UPPER";
            pads[i] = upper ? small : big;
            pads[i + r] = upper ? big : small;
        }
        return pads;
    }

    /// <summary>Returns 4-D view of a 3-D tensor ([N,C,L] → [N,C,1,L]) or the tensor itself.</summary>
    Value To4D(Value x)
    {
        if (x.Rank == 4) return x;
        if (x.Rank == 3) return Rs(x, 0, x.Shape[1].Fixed, 1, x.Shape[2].Fixed);
        throw new NotSupportedException($"only 1-D and 2-D spatial operators are supported (input rank {x.Rank})");
    }

    void Conv(OnnxNode n, Tn?[] ins)
    {
        var x0 = X(ins, 0);
        var wC = ConstIn(ins, 1, "Conv weight");
        var bC = ins.Length > 2 && ins[2] is not null ? ConstIn(ins, 2, "Conv bias") : null;
        bool oneD = x0.Rank == 3;
        var x = To4D(x0);
        long[] wd = oneD ? [wC.Dims[0], wC.Dims[1], 1, wC.Dims[2]] : wC.Dims;
        if (wd.Length != 4) throw new NotSupportedException("only 1-D and 2-D convolutions are supported");
        long groups = n.Int("group", 1);
        long[] Ext(long[]? v, long def) => v is null ? [def, def] : oneD ? [1, v[0]] : v;
        var strides = Ext(n.Ints("strides"), 1);
        var dil = Ext(n.Ints("dilations"), 1);
        long[] kernel = [wd[2], wd[3]];
        long[] pads;
        string autoPad = n.Str("auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER") pads = AutoPad(autoPad, [x.Shape[2].Fixed, x.Shape[3].Fixed], kernel, strides, dil);
        else if (autoPad == "VALID") pads = [0, 0, 0, 0];
        else if (n.Ints("pads") is { } p) pads = oneD ? [0, p[0], 0, p[1]] : p;
        else pads = [0, 0, 0, 0];

        // Fold a directly following BatchNormalization into the weights.
        OnnxNode? bn = FusableBatchNorm(n);
        var weightC = wC.Reshaped(wd);
        OnnxConst? biasC = bC;
        string outName = Nm(n);
        if (bn is not null)
        {
            var (s, t) = BatchNormParams(bn);
            long O = wd[0], per = weightC.Count / O;
            var wsrc = weightC;
            weightC = OnnxConst.Lazy(OnnxType.Float, wd, () =>
            {
                var f = wsrc.F32();
                for (long o = 0; o < O; o++) for (long i = 0; i < per; i++) f[o * per + i] *= s[o];
                return ToBytes(f);
            }, (wC.Name != "" ? wC.Name : outName) + ".bnfold");
            var bsrc = bC;
            biasC = OnnxConst.Lazy(OnnxType.Float, [O], () =>
            {
                var b = bsrc?.F32();
                var r = new float[O];
                for (int o = 0; o < O; o++) r[o] = (b is null ? 0 : b[o] * s[o]) + t[o];
                return ToBytes(r);
            }, outName + ".bnfold_bias");
            _skip.Add(bn);
            outName = bn.Outputs[0];
        }
        var w = WeightValue(weightC, keepHalf: false, outName + ".weight");
        var inputs = new List<Value> { x, w };
        if (biasC is not null) inputs.Add(ConstValue(biasC.Rank == 1 && biasC.ElemType == OnnxType.Float ? biasC : biasC.Reshaped([biasC.Count]), outName + ".bias"));
        var y = Emit("Conv", [.. inputs], Attributes.From(("strides", strides), ("pads", pads), ("dilations", dil), ("groups", groups)), oneD ? outName + ".2d" : outName);
        Heavy();
        if (oneD) y = Rs(y, 0, y.Shape[1].Fixed, y.Shape[3].Fixed);
        _env[bn is not null ? bn.Outputs[0] : n.Outputs[0]] = Tn.Dyn(y);
    }

    static byte[] ToBytes(float[] f)
    {
        var b = new byte[f.Length * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        return b;
    }

    OnnxNode? FusableBatchNorm(OnnxNode conv)
    {
        var o = conv.Outputs[0];
        if (_graphOutputs.Contains(o) || !_consumers.TryGetValue(o, out var cs) || cs.Count != 1) return null;
        var bn = cs[0];
        if (bn.OpType != "BatchNormalization" || bn.Inputs[0] != o || bn.Inputs.Count < 5) return null;
        if (bn.Outputs.Count > 1 && bn.Outputs.Skip(1).Any(x => x != "" && (_usedNames.Contains(x) || _graphOutputs.Contains(x)))) return null;
        for (int i = 1; i < 5; i++) if (!_env.TryGetValue(bn.Inputs[i], out var t) || !t.IsConst) return null;
        return bn;
    }

    (float[] S, float[] T) BatchNormParams(OnnxNode bn)
    {
        var scale = Get(bn.Inputs[1]).C!.F32(); var bias = Get(bn.Inputs[2]).C!.F32();
        var mean = Get(bn.Inputs[3]).C!.F32(); var var_ = Get(bn.Inputs[4]).C!.F32();
        float eps = bn.Float("epsilon", 1e-5f);
        var s = new float[scale.Length]; var t = new float[scale.Length];
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (float)(scale[i] / Math.Sqrt((double)var_[i] + eps));
            t[i] = (float)(bias[i] - (double)mean[i] * s[i]);
        }
        return (s, t);
    }

    void BatchNorm(OnnxNode n, Tn?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        for (int i = 1; i < 5; i++) ConstIn(ins, i, "BatchNormalization parameter");
        var x = X(ins, 0);
        var (s, t) = BatchNormParams(n);
        long[] dims = [s.Length, .. Enumerable.Repeat(1L, Math.Max(0, x.Rank - 2))];
        var sc = OnnxConst.FromFloats(OnnxType.Float, dims, s.Select(v => (double)v).ToArray(), Nm(n) + ".bn_scale");
        var sh = OnnxConst.FromFloats(OnnxType.Float, dims, t.Select(v => (double)v).ToArray(), Nm(n) + ".bn_shift");
        var y = _b.Mul(x, ConstValue(sc, null));
        SetDyn(n, 0, Emit("Add", [y, ConstValue(sh, null)], null, Nm(n)));
    }

    void Pool(OnnxNode n, Tn?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        var x0 = X(ins, 0);
        bool oneD = x0.Rank == 3;
        var x = To4D(x0);
        long[] Ext(long[]? v, long def) => v is null ? [def, def] : oneD ? [1, v[0]] : v;
        var kernel = oneD ? [1, n.Ints("kernel_shape")![0]] : n.Ints("kernel_shape")!;
        var strides = Ext(n.Ints("strides"), 1);
        var dil = Ext(n.Ints("dilations"), 1);
        long[] pads;
        string autoPad = n.Str("auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER") pads = AutoPad(autoPad, [x.Shape[2].Fixed, x.Shape[3].Fixed], kernel, strides, dil);
        else if (autoPad == "VALID") pads = [0, 0, 0, 0];
        else if (n.Ints("pads") is { } p) pads = oneD ? [0, p[0], 0, p[1]] : p;
        else pads = [0, 0, 0, 0];
        bool max = n.OpType == "MaxPool";
        var attrs = Attributes.From(("mode", max ? "max" : "avg"), ("kernel", kernel), ("strides", strides), ("pads", pads), ("dilations", dil),
            ("ceil_mode", n.Int("ceil_mode", 0) != 0), ("count_include_pad", n.Int("count_include_pad", 0) != 0));
        var y = Emit("Pool", [x], attrs, oneD ? Nm(n) + ".2d" : Nm(n));
        if (oneD) y = Rs(y, 0, y.Shape[1].Fixed, y.Shape[3].Fixed);
        SetDyn(n, 0, y);
    }

    void Reduce(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long[]? axes = n.Has("axes") ? n.Ints("axes") : OptConst(ins, 1, "reduction axes")?.Ints();
        bool keep = n.Int("keepdims", 1) != 0;
        if (axes is null || axes.Length == 0)
        {
            if (n.Int("noop_with_empty_axes", 0) != 0 && axes is not null) { SetDyn(n, 0, x); return; }
            axes = Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray();
        }
        axes = axes.Select(a => a < 0 ? a + x.Rank : a).ToArray();
        string name = Nm(n);
        Value y = n.OpType switch
        {
            "ReduceMean" => Emit("Reduce", [x], RA("mean"), name),
            "ReduceSum" => Emit("Reduce", [x], RA("sum"), name),
            "ReduceMax" => Emit("Reduce", [x], RA("max"), name),
            "ReduceMin" => Emit("Reduce", [x], RA("min"), name),
            "ReduceProd" => Emit("Reduce", [x], RA("prod"), name),
            "ReduceL1" => Emit("Reduce", [x], RA("l1"), name),
            "ReduceL2" => Emit("Reduce", [x], RA("l2"), name),
            "ReduceSumSquare" => Emit("Reduce", [x], RA("sumsquare"), name),
            "ReduceLogSumExp" => Emit("Reduce", [x], RA("logsumexp"), name),
            "ReduceLogSum" => Emit("Log", [Emit("Reduce", [x], RA("sum"))], null, name),
            _ => throw new NotSupportedException(n.OpType),
        };
        SetDyn(n, 0, y);
        Attributes RA(string mode) => Attributes.From(("mode", mode), ("axes", axes!), ("keepdims", keep));
    }

    // ================================ matmul =============================================================
    Value Permute(Value x, int[] perm)
    {
        if (perm.SequenceEqual(Enumerable.Range(0, perm.Length))) return x;
        if (Sym(x) && perm[0] != 0) throw new NotSupportedException("permuting the symbolic batch axis away from the outermost position is not supported");
        return _b.Permute(x, perm);
    }

    void Gemm(OnnxNode n, Tn?[] ins)
    {
        float alpha = n.Float("alpha", 1f), beta = n.Float("beta", 1f);
        bool tA = n.Int("transA", 0) != 0, tB = n.Int("transB", 0) != 0;
        var a = X(ins, 0);
        Value y;
        string name = Nm(n);
        if (ins[1]!.IsConst && ins[1]!.C!.Rank == 2)
        {
            if (tA) a = Permute(a, [1, 0]);
            var bc = ins[1]!.C!;
            var w = WeightValue(tB ? bc : bc.Transposed([1, 0]), keepHalf: true, name + ".weight");
            y = Emit("MatMul", [a, w], null, name);
            if (alpha != 1f) y = _b.Scale(y, alpha);
        }
        else
        {
            y = Emit("BatchMatMul", [a, X(ins, 1)], Attributes.From(("trans_a", tA), ("trans_b", tB), ("alpha", alpha)), name);
        }
        Heavy();
        if (ins.Length > 2 && ins[2] is { } c)
        {
            Value cv;
            if (c.IsConst) cv = ConstValue(beta == 1f ? c.C! : OnnxConst.FromFloats(OnnxType.Float, c.C!.Dims, c.C.Floats().Select(v => v * beta).ToArray(), c.C.Name + ".beta"), name + ".bias");
            else cv = beta == 1f ? c.V! : _b.Scale(c.V!, beta);
            y = Emit("Add", [y, cv], null, name + ".biased");
        }
        SetDyn(n, 0, y);
    }

    void MatMul(OnnxNode n, Tn?[] ins)
    {
        var aT = ins[0]!; var bT = ins[1]!;
        string name = Nm(n);
        Value y;
        if (!aT.IsConst && bT.IsConst && bT.C!.Rank == 2 && aT.V!.Rank >= 2)
        {
            var w = WeightValue(bT.C!.Transposed([1, 0]), keepHalf: true, name + ".weight");
            y = Emit("MatMul", [aT.V!, w], null, name);
        }
        else
        {
            var a = AsValue(aT); var b = AsValue(bT);
            if (a.Rank < 2 || b.Rank < 2) throw new NotSupportedException("MatMul with 1-D operands is not supported");
            y = Emit("BatchMatMul", [a, b], null, name);
        }
        Heavy();
        SetDyn(n, 0, y);
    }

    // ================================ shape manipulation ===================================================
    /// <summary>ONNX Reshape semantics (0 = copy, -1 = infer; <see cref="OnnxConst.SymBatch"/> = batch size) on top of Sharplight's.</summary>
    Value ReshapeOnnx(Value x, long[] req, bool allowZero, string? name = null)
    {
        bool srcSym = Sym(x);
        var r = new long[req.Length];
        for (int k = 0; k < req.Length; k++)
        {
            long v = req[k];
            if (v == OnnxConst.SymBatch)
            {
                if (!srcSym || k != 0) throw new NotSupportedException("the target shape mentions the symbolic batch size somewhere other than the outermost dim of a batched tensor");
                r[k] = 0; continue;
            }
            if (v == 0 && !allowZero)
            {
                if (k >= x.Rank) throw new InvalidDataException("Reshape: 0 beyond the input rank.");
                if (k == 0 && srcSym) { r[0] = 0; continue; }
                v = x.Shape[k].Fixed;
            }
            r[k] = v;
        }
        if (srcSym)
        {
            if (r.Length == 0) throw new NotSupportedException("reshaping a batched tensor to a scalar");
            if (r[0] == -1)
            {
                long rest = 1;
                for (int k = 1; k < r.Length; k++) { if (r[k] == -1) throw new NotSupportedException("two inferred dims"); rest *= r[k]; }
                if (rest != x.Shape.InnerElements) throw new NotSupportedException($"reshape of batched {x.Shape} to [{string.Join(",", req)}] would merge the batch dimension into another dim");
                r[0] = 0;
            }
            else if (r[0] != 0) throw new NotSupportedException($"reshape of batched {x.Shape} fixes the batch dimension to {r[0]}");
        }
        else if (req.Any(v => v == OnnxConst.SymBatch)) throw new NotSupportedException("the target shape mentions the symbolic batch size but the input is not batched");
        return Emit("Reshape", [x], Attributes.From(("shape", r)), name);
    }

    void Flatten(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        int axis = (int)n.Int("axis", 1); if (axis < 0) axis += x.Rank;
        bool sym = Sym(x);
        if (sym && axis != 1) throw new NotSupportedException($"Flatten(axis={axis}) of a batched tensor would merge or hide the batch dimension");
        long inner = 1;
        for (int i = axis; i < x.Rank; i++) inner *= x.Shape[i].Fixed;
        long outer = 1;
        if (!sym) for (int i = 0; i < axis; i++) outer *= x.Shape[i].Fixed;
        SetDyn(n, 0, Emit("Reshape", [x], Attributes.From(("shape", L(sym ? 0 : outer, inner))), Nm(n)));
    }

    void SqueezeUnsqueeze(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        bool sym = Sym(x);
        var axesArr = n.Has("axes") ? n.Ints("axes") : OptConst(ins, 1, "axes")?.Ints();
        var dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();   // 0 marks the symbolic outer dim
        long[] outDims;
        if (n.OpType == "Unsqueeze")
        {
            if (axesArr is null) throw new InvalidDataException("Unsqueeze without axes");
            outDims = OnnxFold.UnsqueezeDims(dims, axesArr);
            if (sym && outDims[0] != 0) throw new NotSupportedException("Unsqueeze at axis 0 of a batched tensor would move the batch dimension");
        }
        else
        {
            if (sym && axesArr is not null && axesArr.Any(a => a == 0 || a == -x.Rank)) throw new NotSupportedException("Squeeze of the batch dimension");
            outDims = OnnxFold.SqueezeDims(sym ? dims.Select((d, i) => i == 0 ? -7 : d).ToArray() : dims, axesArr);
            if (sym) outDims = outDims.Select(d => d == -7 ? 0 : d).ToArray();
            if (sym && axesArr is null && outDims[0] != 0) throw new NotSupportedException("Squeeze moved the batch dimension");
        }
        SetDyn(n, 0, Emit("Reshape", [x], Attributes.From(("shape", outDims)), Nm(n)));
    }

    void Concat(OnnxNode n, Tn?[] ins)
    {
        var vals = ins.Select((t, i) => X(ins, i, Nm(n) + $".in{i}")).ToList();
        int axis = (int)n.Int("axis", 0); if (axis < 0) axis += vals[0].Rank;
        if (axis == 0 && Sym(vals[0])) throw new NotSupportedException("Concat along the symbolic batch axis");
        if (vals.Count == 1) { SetDyn(n, 0, vals[0]); return; }
        var cur = vals;
        while (cur.Count > 64) cur = [_b.Concat(axis, cur.Take(64).ToArray()), .. cur.Skip(64)];
        SetDyn(n, 0, Emit("Concat", [.. cur], Attributes.From(("axis", axis)), Nm(n)));
    }

    Value SliceAxis(Value x, int axis, long start, long length)
    {
        if (start == 0 && length == x.Shape[axis].Fixed) return x;
        return _b.Slice(x, axis, start, length);
    }

    void Slice(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long[] starts, ends; long[]? axes, steps = null;
        if (n.Has("starts")) { starts = n.Ints("starts")!; ends = n.Ints("ends")!; axes = n.Ints("axes"); }
        else
        {
            starts = ConstIn(ins, 1, "Slice starts").Ints(); ends = ConstIn(ins, 2, "Slice ends").Ints();
            axes = OptConst(ins, 3, "Slice axes")?.Ints(); steps = OptConst(ins, 4, "Slice steps")?.Ints();
        }
        axes ??= Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
        var y = x;
        for (int i = 0; i < axes.Length; i++)
        {
            int ax = (int)(axes[i] < 0 ? axes[i] + x.Rank : axes[i]);
            long step = steps is null ? 1 : steps[i];
            if (step == 0) throw new InvalidDataException("Slice with step 0");
            if (step < 0)
            {
                if (ax == 0 && Sym(x)) throw new NotSupportedException("slicing the symbolic batch axis");
                y = ReverseSlice(y, ax, starts[i], ends[i], step);
                continue;
            }
            if (ax == 0 && Sym(x))
            {
                if (starts[i] == 0 && ends[i] >= int.MaxValue) continue;
                throw new NotSupportedException("slicing the symbolic batch axis");
            }
            var (s, l) = OnnxFold.SliceRange(starts[i], ends[i], x.Shape[ax].Fixed);
            y = step == 1 ? SliceAxis(y, ax, s, l) : StridedSlice(y, ax, s, l, step);
        }
        SetDyn(n, 0, ReferenceEquals(y, x) ? Emit("Reshape", [x], Attributes.From(("shape", x.Shape.Dims.Select((d, i) => d.IsSymbolic ? 0L : d.Fixed).ToArray())), Nm(n)) : y);
    }

    /// <summary>Negative-step slice as a gather with constant indices.</summary>
    Value ReverseSlice(Value x, int ax, long start, long end, long step)
    {
        long dim = x.Shape[ax].Fixed;
        if (start < 0) start += dim;
        start = Math.Clamp(start, 0, dim - 1);
        if (end < 0) end += dim;
        end = Math.Clamp(end, -1, dim - 1);
        var idx = new List<long>();
        for (long k = start; k > end; k += step) idx.Add(k);
        if (idx.Count == 0) throw new NotSupportedException("empty Slice result");
        return Emit("GatherAxis", [x, Idx(Tn.Const(OnnxConst.FromInts(OnnxType.Int64, [idx.Count], idx.ToArray())))], Attributes.From(("axis", (long)ax)));
    }

    /// <summary>x[start : start+length : step] along <paramref name="ax"/> as Slice + Reshape([..., groups, step, ...]) + Slice(0) (+ one trailing element).</summary>
    Value StridedSlice(Value x, int ax, long start, long length, long step)
    {
        long full = length / step, rem = length % step;
        long[] dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();
        Value? main = null;
        if (full > 0)
        {
            var region = SliceAxis(x, ax, start, full * step);
            var split = dims.Take(ax).Append(full).Append(step).Concat(dims.Skip(ax + 1)).ToArray();
            var picked = _b.Slice(_b.Reshape(region, split), ax + 1, 0, 1);
            main = _b.Reshape(picked, dims.Take(ax).Append(full).Concat(dims.Skip(ax + 1)).ToArray());
        }
        if (rem == 0) return main!;
        var tail = _b.Slice(x, ax, start + full * step, 1);
        return main is null ? tail : _b.Concat(ax, main, tail);
    }

    void Split(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        int axis = (int)n.Int("axis", 0); if (axis < 0) axis += x.Rank;
        if (axis == 0 && Sym(x)) throw new NotSupportedException("Split along the symbolic batch axis");
        var sizes = n.Has("split") ? n.Ints("split") : OptConst(ins, 1, "Split sizes")?.Ints();
        int outs = n.Outputs.Count;
        sizes ??= Enumerable.Repeat(x.Shape[axis].Fixed / outs, outs).ToArray();
        long start = 0;
        for (int i = 0; i < outs; i++)
        {
            var piece = SliceAxis(x, axis, start, sizes[i]);
            if (ReferenceEquals(piece, x)) piece = Emit("Reshape", [x], Attributes.From(("shape", x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray())));
            if (n.Outputs[i] != "") SetDyn(n, i, piece);
            start += sizes[i];
        }
    }

    void Gather(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        if (!ins[1]!.IsConst)
        {
            SetDyn(n, 0, Emit("GatherAxis", [x, Idx(ins[1]!)], Attributes.From(("axis", n.Int("axis", 0))), Nm(n)));
            return;
        }
        var idxC = ConstIn(ins, 1, "Gather indices");
        int axis = (int)n.Int("axis", 0); if (axis < 0) axis += x.Rank;
        bool sym = Sym(x);
        if (axis == 0 && sym) throw new NotSupportedException("Gather along the symbolic batch axis");
        long dimA = x.Shape[axis].Fixed;
        var idx = idxC.Ints().Select(v => v < 0 ? v + dimA : v).ToArray();
        if (idx.Any(v => v < 0 || v >= dimA)) throw new InvalidDataException("Gather index out of range.");
        // output shape: x.dims[:axis] + idx.dims + x.dims[axis+1:]
        var outDims = x.Shape.Dims.Take(axis).Select(d => d.IsSymbolic ? 0L : d.Fixed)
            .Concat(idxC.Dims).Concat(x.Shape.Dims.Skip(axis + 1).Select(d => d.Fixed)).ToArray();
        Value y;
        if (idx.Length == 0) throw new NotSupportedException("Gather with empty indices");
        bool contiguous = true;
        for (int i = 1; i < idx.Length; i++) if (idx[i] != idx[i - 1] + 1) contiguous = false;
        if (contiguous) y = SliceAxis(x, axis, idx[0], idx.Length);
        else
        {
            if (idx.Length > 4096) throw new NotSupportedException($"Gather with {idx.Length} indices");
            var pieces = idx.Select(i => SliceAxis(x, axis, i, 1)).ToList();
            while (pieces.Count > 64) pieces = pieces.Chunk(64).Select(c => c.Length == 1 ? c[0] : _b.Concat(axis, c)).ToList();
            y = pieces.Count == 1 ? pieces[0] : _b.Concat(axis, pieces.ToArray());
        }
        SetDyn(n, 0, Emit("Reshape", [y], Attributes.From(("shape", outDims)), Nm(n)));
    }

    /// <summary>A zero tensor [N,1,...,1] with the dynamic batch size, used to materialize broadcasts.</summary>
    Value BatchZeros(int rank)
    {
        if (_batchZero is null)
        {
            var inp = _firstInput ?? throw new NotSupportedException("broadcasting a constant to the batch size needs a floating point graph input");
            var v = inp;
            for (int a = 1; a < inp.Rank; a++) v = _b.Slice(v, a, 0, 1);
            _batchZero = _b.Scale(v, 0f, 0f);
        }
        var shape = new long[rank]; for (int i = 1; i < rank; i++) shape[i] = 1;
        return Rs(_batchZero, shape);
    }

    void Expand(OnnxNode n, Tn?[] ins)
    {
        var xt = ins[0]!;
        var target = ConstIn(ins, 1, "Expand shape").Ints();
        var x = AsValue(xt);
        int rank = Math.Max(x.Rank, target.Length);
        var xd = Enumerable.Repeat(1L, rank - x.Rank).Concat(x.Shape.Dims.Select(d => d.IsSymbolic ? OnnxConst.SymBatch : d.Fixed)).ToArray();
        var td = Enumerable.Repeat(1L, rank - target.Length).Concat(target).ToArray();
        var res = new long[rank];
        bool changed = false;
        for (int i = 0; i < rank; i++)
        {
            if (td[i] == OnnxConst.SymBatch || xd[i] == OnnxConst.SymBatch) { res[i] = OnnxConst.SymBatch; if (xd[i] != OnnxConst.SymBatch) changed = true; }
            else if (td[i] == xd[i] || td[i] == 1) res[i] = xd[i];
            else if (xd[i] == 1) { res[i] = td[i]; changed = true; }
            else throw new InvalidDataException("Expand: incompatible shapes.");
        }
        if (res.Count(v => v == OnnxConst.SymBatch) > 1 || (res.Any(v => v == OnnxConst.SymBatch) && res[0] != OnnxConst.SymBatch))
            throw new NotSupportedException("Expand of the batch dimension to a position other than the outermost");
        if (!changed && x.Rank == rank) { _env[n.Outputs[0]] = xt; return; }
        // y = x + zeros(target): the zeros come from a batch-sized zero tensor and (if needed) a static zero constant.
        bool hasBatch = res[0] == OnnxConst.SymBatch;
        var zeroParts = new List<Value>();
        var innerDims = res.Skip(hasBatch ? 1 : 0).ToArray();
        bool needInner = false;
        for (int i = hasBatch ? 1 : 0; i < rank; i++) if (xd[i] != res[i]) needInner = true;
        Value y = x;
        if (x.Rank < rank) y = Emit("Reshape", [x], Attributes.From(("shape", xd.Select((d, i) => d == OnnxConst.SymBatch ? 0L : d).ToArray())));
        if (hasBatch && xd[0] != OnnxConst.SymBatch) y = Emit("Add", [y, BatchZeros(rank)]);
        if (needInner)
        {
            var zd = (hasBatch ? new long[] { 1 } : []).Concat(innerDims).ToArray();
            var z = _b.Constant($"{Nm(n)}.zeros", DataType.F32, new Shape(zd.Select(d => (Dim)d).ToArray()), new byte[zd.Aggregate(1L, (a, b) => a * b) * 4]);
            y = Emit("Add", [y, z]);
        }
        SetDyn(n, 0, y);
    }

    // ================================ normalization / misc =================================================
    void Softmax(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        int rank = x.Rank;
        int axis = (int)n.Int("axis", Opset >= 13 ? -1 : 1); if (axis < 0) axis += rank;
        Value y;
        bool lg = n.OpType == "LogSoftmax";
        Value Sm(Value v) => lg ? _b.Emit1("Softmax", [v], Attributes.From(("log", true))) : _b.Emit1("Softmax", [v]);
        if (axis == rank - 1) y = Sm(x);
        else if (Opset >= 13)
        {
            if (axis == 0 && Sym(x)) throw new NotSupportedException("Softmax over the symbolic batch axis");
            var perm = Enumerable.Range(0, rank).ToArray(); (perm[axis], perm[rank - 1]) = (perm[rank - 1], perm[axis]);
            y = Permute(Sm(Permute(x, perm)), perm);
        }
        else
        {
            // Pre-13: coerce to 2-D at 'axis' (softmax over all trailing dims).
            if (axis == 0 && Sym(x)) throw new NotSupportedException("Softmax over the symbolic batch axis");
            bool sym = Sym(x);
            var dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();
            long inner = 1; for (int i = axis; i < rank; i++) inner *= dims[i];
            long[] flat = [.. dims[..axis], inner];
            var s = Sm(Rs(x, flat));
            y = Rs(s, dims);
        }
        SetDyn(n, 0, y);
        _ = Nm(n);
    }

    void LayerNorm(OnnxNode n, Tn?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        var x = X(ins, 0);
        int axis = (int)n.Int("axis", -1); if (axis < 0) axis += x.Rank;
        if (axis == 0 && Sym(x)) throw new NotSupportedException("LayerNormalization over the batch axis");
        var sc = ConstIn(ins, 1, "LayerNormalization scale");
        var bi = OptConst(ins, 2, "LayerNormalization bias");
        long d = 1; for (int i = axis; i < x.Rank; i++) d *= x.Shape[i].Fixed;
        var xin = x;
        bool flatten = axis != x.Rank - 1;
        if (flatten)
        {
            long[] flat = [.. x.Shape.Dims.Take(axis).Select(q => q.IsSymbolic ? 0L : q.Fixed), d];
            xin = Rs(x, flat);
        }
        var ws = new List<Value> { xin, ConstValue(sc.Reshaped([d]), Nm(n) + ".scale") };
        if (bi is not null) ws.Add(ConstValue(bi.Reshaped([d]), Nm(n) + ".bias"));
        var y = Emit("LayerNorm", [.. ws], Attributes.From(("eps", n.Float("epsilon", 1e-5f))), flatten ? null : Nm(n));
        if (flatten) y = Emit("Reshape", [y], Attributes.From(("shape", x.Shape.Dims.Select(q => q.IsSymbolic ? 0L : q.Fixed).ToArray())), Nm(n));
        SetDyn(n, 0, y);
    }

    void InstanceNorm(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        var sc = ConstIn(ins, 1, "InstanceNormalization scale"); var bi = ConstIn(ins, 2, "InstanceNormalization bias");
        SetDyn(n, 0, Emit("GroupNorm", [x, ConstValue(sc, Nm(n) + ".scale"), ConstValue(bi, Nm(n) + ".bias")],
            Attributes.From(("groups", x.Shape[1].Fixed), ("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    void Pad(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long[] pads = n.Has("pads") ? n.Ints("pads")! : ConstIn(ins, 1, "Pad pads").Ints();
        float value = n.Has("value") ? n.Float("value", 0) : OptConst(ins, 2, "Pad constant_value") is { } cv ? cv.ScalarF32() : 0f;
        if (OptConst(ins, 3, "Pad axes") is { } axesC)
        {
            var full = new long[2 * x.Rank];
            var ax = axesC.Ints();
            for (int i = 0; i < ax.Length; i++) { int a = (int)(ax[i] < 0 ? ax[i] + x.Rank : ax[i]); full[a] = pads[i]; full[a + x.Rank] = pads[i + ax.Length]; }
            pads = full;
        }
        string mode = n.Str("mode", "constant");
        if (mode is not ("constant" or "reflect" or "edge")) throw new NotSupportedException($"Pad mode '{mode}'");
        if (pads.All(p => p == 0)) { SetDyn(n, 0, x); return; }
        SetDyn(n, 0, Emit("Pad", [x], Attributes.From(("pads", pads), ("mode", mode), ("value", value)), Nm(n)));
    }

    void Resize(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        if (x.Rank != 4) throw new NotSupportedException("only 4-D (NCHW) Resize is supported");
        bool upsample = n.OpType == "Upsample";
        OnnxConst? scales = null, sizes = null;
        if (upsample || (ins.Length == 2)) scales = n.Has("scales") ? OnnxConst.FromFloats(OnnxType.Float, [n.Floats("scales")!.Length], n.Floats("scales")!.Select(v => (double)v).ToArray()) : OptConst(ins, 1, "Resize scales");
        else { scales = OptConst(ins, 2, "Resize scales"); sizes = OptConst(ins, 3, "Resize sizes"); }
        if (scales is { Count: 0 }) scales = null;
        long h = x.Shape[2].Fixed, w = x.Shape[3].Fixed;
        long[] size;
        if (sizes is not null)
        {
            var s = sizes.Ints();
            if (s.Length != 4 || s[1] != x.Shape[1].Fixed || (!Sym(x) && s[0] != x.Shape[0].Fixed)) throw new NotSupportedException("Resize may only change the spatial dimensions");
            size = [s[2], s[3]];
        }
        else if (scales is not null)
        {
            var s = scales.Floats();
            if (s.Length != 4 || s[0] != 1 || s[1] != 1) throw new NotSupportedException("Resize may only scale the spatial dimensions");
            size = [(long)MathF.Floor((float)h * (float)s[2]), (long)MathF.Floor((float)w * (float)s[3])];
        }
        else throw new NotSupportedException("Resize needs constant scales or sizes");
        string mode = n.Str("mode", "nearest");
        if (mode == "bilinear") mode = "linear";
        if (mode == "bicubic") mode = "cubic";
        if (mode is not ("nearest" or "linear" or "cubic")) throw new NotSupportedException($"Resize mode '{mode}'");
        string coord = n.Str("coordinate_transformation_mode", upsample || (!upsample && ins.Length == 2) ? "asymmetric" : "half_pixel");
        if (coord is not ("half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel")) throw new NotSupportedException($"Resize coordinate_transformation_mode '{coord}'");
        if (n.Int("antialias", 0) != 0) throw new NotSupportedException("antialiased Resize");
        if (n.Int("exclude_outside", 0) != 0) throw new NotSupportedException("Resize exclude_outside");
        string nearest = n.Str("nearest_mode", upsample ? "floor" : "round_prefer_floor");
        if (mode == "cubic")
        {
            double[]? sc = scales is not null && sizes is null ? [scales.Floats()[2], scales.Floats()[3]] : null;
            SetDyn(n, 0, ResizeCubic(n, x, size, sc, coord));
            return;
        }
        if (size[0] == h && size[1] == w && mode == "nearest") { SetDyn(n, 0, x); return; }
        SetDyn(n, 0, Emit("Resize", [x], Attributes.From(("size", size), ("mode", mode), ("coord", coord), ("nearest_mode", nearest)), Nm(n)));
    }
}
