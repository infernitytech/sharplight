using Sharplight.Graph;

namespace Sharplight.Onnx;

// Quantization: Quantize/DequantizeLinear, the integer matmul/conv family and the com.microsoft QLinear* operators. Quantized activations are
// carried as f32 holding integer values; scales and zero points may be compile-time constants or runtime values (DynamicQuantizeLinear).
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_quantContrib =
    [
        "QLinearAdd", "QLinearMul", "QLinearGlobalAveragePool", "QLinearAveragePool", "QLinearSigmoid", "QLinearLeakyRelu", "QLinearConcat", "QLinearSoftmax",
    ];

    /// <summary>A scale or zero point as an f32 value broadcastable against a tensor (constant, or a runtime scalar).</summary>
    private Value QParam(TranslatedTensor t, string name) => t.IsConst ? ConstValue(t.C!.CastTo(OnnxType.Float).Count == 1 ? t.C.CastTo(OnnxType.Float).Reshaped([1]) : t.C.CastTo(OnnxType.Float), name) : AsValue(t);

    /// <summary>(q - zp) * scale.</summary>
    private Value Deq(TranslatedTensor x, TranslatedTensor scale, TranslatedTensor? zp, string name)
    {
        var v = AsValue(x);
        bool zeroZp = zp is null || (zp.IsConst && zp.C!.Floats().All(z => z == 0));
        if (!zeroZp)
        {
            v = _b.Sub(v, QParam(zp!, name + ".zp"));
        }

        if (scale.IsConst && scale.C!.Count == 1)
        {
            return _b.Scale(v, scale.C.ScalarF32());
        }

        return _b.Mul(v, QParam(scale, name + ".scale"));
    }

    /// <summary>clip(round(y / scale) + zp, range of the zero-point type).</summary>
    private Value Req(Value y, TranslatedTensor scale, TranslatedTensor? zp, string name)
    {
        var q = Emit("Round", [_b.Binary("Div", y, QParam(scale, name + ".ys"))]);   // a true division (not * 1/s) so ties round like onnxruntime
        int et = zp is { IsConst: true } ? zp.C!.ElemType : OnnxType.UInt8;
        if (zp is not null)
        {
            q = _b.Add(q, QParam(zp, name + ".yzp"));
        }

        var (lo, hi) = OnnxFold.Range(et);
        return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
    }

    // ---- quantization ----------------------------------------------------------------------------------------
    /// <summary>Dequantizes a constant (per-tensor/per-axis zero point and scale) to f32.</summary>
    private static OnnxConst DequantConst(OnnxConst x, OnnxConst scale, OnnxConst? zp, int axis = 0) => OnnxFold.Dequantize(x, scale, zp, axis);

    private Value ShiftZp(TranslatedTensor a, TranslatedTensor? zpT, string nm)
    {
        if (zpT is { IsConst: false })
        {
            return _b.Sub(AsValue(a), AsValue(zpT));   // runtime zero point (DynamicQuantizeLinear)
        }

        return ShiftZp(a, zpT?.C, nm);
    }

    private Value ShiftZp(TranslatedTensor a, OnnxConst? zp, string nm)
    {
        var v = AsValue(a);
        if (zp is null || zp.Floats().All(z => z == 0))
        {
            return v;
        }

        return _b.Sub(v, ConstValue(zp.Count == 1 ? zp.CastTo(OnnxType.Float).Reshaped([1]) : throw new NotSupportedException("per-row activation zero points"), nm + ".azp"));
    }

    /// <summary>
    /// clip(round(acc · m) + zp) for an integer accumulator, with onnxruntime's f32 multipliers m = a_scale · w_scale / y_scale (one, or one
    /// per output channel along <paramref name="channelAxis"/>), so results match its integer kernels bit for bit.
    /// </summary>
    private Value Requant(Value acc, float[] multipliers, int channelAxis, OnnxConst? yzp, string name)
    {
        Value scaled;
        if (multipliers.Length == 1)
        {
            scaled = _b.Scale(acc, multipliers[0]);
        }
        else
        {
            var dims = Enumerable.Repeat(1L, acc.Rank - channelAxis).ToArray();
            dims[0] = multipliers.Length;
            scaled = _b.Mul(acc, ConstValue(OnnxConst.FromFloats(OnnxType.Float, dims, multipliers.Select(m => (double)m).ToArray()), name + ".m"));
        }

        var q = Emit("Round", [scaled]);
        int et = yzp?.ElemType ?? OnnxType.UInt8;
        if (yzp is not null)
        {
            q = _b.Scale(q, 1f, yzp.ScalarF32());
        }

        var (lo, hi) = OnnxFold.Range(et);
        return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
    }

    /// <summary>
    /// y = clip(round(x / s) + zp, 0, 255) with s = (max(x,0) - min(x,0)) / 255 (1 / 255 when both are 0) and zp = clip(round(-min(x,0) / s), 0, 255),
    /// over the whole tensor.
    /// </summary>
    private void DynamicQuantize(OnnxNode n, TranslatedTensor?[] ins)
    {
        var (y, scale, zeroPoint) = DynamicQuantizeValues(X(ins, 0), Nm(n));
        SetDyn(n, 0, y);
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, Rs(scale, 1));
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, Rs(zeroPoint, 1));
        }
    }

    /// <summary>DynamicQuantizeLinear of <paramref name="x"/>: the uint8 values (as f32), the scale and the zero point (both of rank x).</summary>
    private (Value Q, Value Scale, Value ZeroPoint) DynamicQuantizeValues(Value x, string? name)
    {
        var all = Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray();
        var maxAttributes = Attributes.From(("mode", "max"), ("axes", all), ("keepdims", true));
        var minAttributes = Attributes.From(("mode", "min"), ("axes", all), ("keepdims", true));
        var hi = Emit("Clip", [Emit("Reduce", [x], maxAttributes)], Attributes.From(("min", 0f)));
        var lo = Emit("Clip", [Emit("Reduce", [x], minAttributes)], Attributes.From(("max", 0f)));
        // scale = (max - min) / 255, or 1 / 255 for an all-zero tensor (min = max = 0).
        var range = _b.Sub(hi, lo);
        range = _b.Add(range, Emit("Equal", [range, ConstValue(OnnxConst.Scalar(0f).Reshaped([1]), "zero")]));
        var scale = _b.Binary("Div", range, ConstValue(OnnxConst.Scalar(255f).Reshaped([1]), "qmax"));
        var byteRange = Attributes.From(("min", 0f), ("max", 255f));
        var zeroPoint = Emit("Clip", [Emit("Round", [_b.Binary("Div", _b.Scale(lo, -1f), scale)])], byteRange);
        var y = Emit("Clip", [_b.Add(Emit("Round", [_b.Binary("Div", x, scale)]), zeroPoint)], byteRange, name);
        return (y, scale, zeroPoint);
    }

    private void MatMulInteger(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var a = ShiftZp(ins[0]!, Opt(ins, 2), name);
        if (!ins[1]!.IsConst)
        {
            // Runtime B (e.g. attention scores of dynamically quantized activations): subtract its zero point and multiply as f32.
            var shifted = Opt(ins, 3) is { } zeroPoint ? Typed(_b.Sub(AsValue(ins[1]!), AsValue(zeroPoint)), OnnxType.Int32) : ins[1]!;
            _env[n.Outputs[0]] = Synthetic("MatMul", [TranslatedTensor.Dyn(a), shifted]);
            return;
        }

        var bConst = ConstIn(ins, 1, "MatMulInteger B");
        var bZeroPoint = OptConst(ins, 3, "b zero point");
        var b = DequantConst(bConst, OnnxConst.Scalar(1f), bZeroPoint, bConst.Rank - 1);   // per-column zero points when bZeroPoint is a vector
        SetDyn(n, 0, MatMulWeightOrBatch(a, b, name));
    }

    private void QLinearMatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        if (Enumerable.Range(1, 7).Any(i => Opt(ins, i) is { IsConst: false }))
        {
            QLinearMatMulRuntime(n, ins);
            return;
        }

        var a = ShiftZp(ins[0]!, OptConst(ins, 2, "a_zp"), name);
        var b = DequantConst(ConstIn(ins, 3, "B"), OnnxConst.Scalar(1f), OptConst(ins, 5, "b_zp"), 1);
        var acc = MatMulWeightOrBatch(a, b, name + ".mm");
        var multipliers = Multipliers(ConstIn(ins, 1, "a_scale").ScalarF32(), ConstIn(ins, 4, "b_scale"), ConstIn(ins, 6, "y_scale").ScalarF32());
        SetDyn(n, 0, Requant(acc, multipliers, acc.Rank - 1, OptConst(ins, 7, "y_zp"), name));
    }

    /// <summary>
    /// QLinearMatMul with runtime B, scales or zero points: (a - a_zp) · (b - b_zp) as an f32 matmul of exact integers, requantized with
    /// the multiplier a_scale · b_scale / y_scale computed at run time.
    /// </summary>
    private void QLinearMatMulRuntime(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var a = Typed(ShiftZp(ins[0]!, Opt(ins, 2), name), OnnxType.Int32);
        var b = Opt(ins, 5) is { } bZeroPoint ? Typed(_b.Sub(AsValue(ins[3]!), QParam(bZeroPoint, name + ".bzp")), OnnxType.Int32) : ins[3]!;
        var acc = AsValue(Synthetic("MatMul", [a, b]));
        var multiplier = _b.Binary("Div", _b.Mul(QParam(ins[1]!, name + ".as"), QParam(ins[4]!, name + ".bs")), QParam(ins[6]!, name + ".ys"));
        var q = Emit("Round", [_b.Mul(acc, multiplier)]);
        var yZeroPoint = Opt(ins, 7);
        if (yZeroPoint is not null)
        {
            q = _b.Add(q, QParam(yZeroPoint, name + ".yzp"));
        }

        int type = yZeroPoint is not null ? ElemTypeOf(yZeroPoint) : OnnxType.UInt8;
        var (lo, hi) = OnnxFold.Range(type);
        _env[n.Outputs[0]] = Typed(Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name), type);
    }

    /// <summary>onnxruntime's requantization multipliers a_scale · w_scale / y_scale, evaluated in f32 in the same order.</summary>
    private static float[] Multipliers(float inputScale, OnnxConst weightScale, float outputScale)
        => weightScale.Floats().Select(w => inputScale * (float)w / outputScale).ToArray();

    /// <summary>
    /// ConvInteger and QLinearConv: a regular f32 Conv over the zero-point-shifted integer values (exact integer accumulation while it stays
    /// below 2^24) and, for QLinearConv, onnxruntime's requantization of the accumulator.
    /// </summary>
    private void QuantizedConv(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        bool linear = n.OpType == "QLinearConv";
        var x = linear ? ShiftZp(ins[0]!, OptConst(ins, 2, "x_zp"), name) : ShiftZp(ins[0]!, Opt(ins, 2), name);
        var weightConst = ConstIn(ins, linear ? 3 : 1, "W");
        var weightZeroPoint = OptConst(ins, linear ? 5 : 3, "w_zp");
        var weight = DequantConst(weightConst, OnnxConst.Scalar(1f), weightZeroPoint, 0);
        var bias = linear && ins.Length > 8 && ins[8] is not null ? ConstIn(ins, 8, "B").CastTo(OnnxType.Float).Reshaped([weight.Dims[0]]) : null;

        var convInputs = new TranslatedTensor?[] { TranslatedTensor.Dyn(x), TranslatedTensor.Const(weight.Reshaped(weight.Dims, name + ".w")), bias is null ? null : TranslatedTensor.Const(bias) };
        var conv = new OnnxNode { OpType = "Conv", Name = n.Name, Inputs = ["x", "w", bias is null ? "" : "b"], Outputs = [name + ".conv"], Attributes = n.Attributes };
        Conv(conv, convInputs);
        var y = _env[name + ".conv"].V!;
        if (linear)
        {
            var multipliers = Multipliers(ConstIn(ins, 1, "x_scale").ScalarF32(), ConstIn(ins, 4, "w_scale"), ConstIn(ins, 6, "y_scale").ScalarF32());
            y = Requant(y, multipliers, 1, OptConst(ins, 7, "y_zp"), name);
        }

        SetDyn(n, 0, y);
    }

    /// <summary>Quantized com.microsoft operators (QLinearAdd, QLinearSigmoid, ...): dequantize, apply the f32 op, requantize.</summary>
    private void TranslateQuantContrib(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        switch (n.OpType)
        {
            case "QLinearAdd" or "QLinearMul":
            {
                var a = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".a");
                var b = Deq(ins[3]!, ins[4]!, Opt(ins, 5), name + ".b");
                var r = n.OpType == "QLinearAdd" ? _b.Add(a, b) : _b.Mul(a, b);
                SetDyn(n, 0, Req(r, ins[6]!, Opt(ins, 7), name));
                break;
            }
            case "QLinearSigmoid":
            {
                var x = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x");
                SetDyn(n, 0, Req(_b.Sigmoid(x), ins[3]!, Opt(ins, 4), name));
                break;
            }
            case "QLinearLeakyRelu":
            {
                var x = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x");
                var y = Emit("LeakyRelu", [x], Attributes.From(("alpha", n.Float("alpha", 0.01f))));
                SetDyn(n, 0, Req(y, ins[3]!, Opt(ins, 4), name));
                break;
            }
            case "QLinearSoftmax":
            {
                var softmax = new OnnxNode { OpType = "Softmax", Name = n.Name, Inputs = ["x"], Outputs = [name + ".sm"], Attributes = n.Attributes };
                Dynamic(softmax, [TranslatedTensor.Dyn(Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x"))]);
                SetDyn(n, 0, Req(_env[name + ".sm"].V!, ins[3]!, Opt(ins, 4), name));
                break;
            }
            case "QLinearGlobalAveragePool" or "QLinearAveragePool":
                QLinearPool(n, ins);
                break;
            case "QLinearConcat":
            {
                var parts = new List<Value>();
                for (int i = 2; i + 2 < ins.Length; i += 3)
                {
                    parts.Add(Deq(ins[i]!, ins[i + 1]!, Opt(ins, i + 2), name + $".in{i}"));
                }

                var concat = Emit("Concat", [.. parts], Attributes.From(("axis", n.Int("axis", 1))));
                SetDyn(n, 0, Req(concat, ins[0]!, Opt(ins, 1), name));
                break;
            }
            default:
                throw new NotSupportedException(n.OpType);
        }
    }

    private void QLinearPool(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        bool channelsLast = n.Int("channels_last", 0) != 0;
        var x = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x");
        if (channelsLast)
        {
            x = Permute(x, [0, 3, 1, 2]);
        }

        string poolOp = n.OpType == "QLinearGlobalAveragePool" ? "GlobalAveragePool" : "AveragePool";
        var pool = new OnnxNode { OpType = poolOp, Name = n.Name, Inputs = ["x"], Outputs = [name + ".pool"], Attributes = n.Attributes };
        Dynamic(pool, [TranslatedTensor.Dyn(x)]);
        var y = _env[name + ".pool"].V!;
        if (channelsLast)
        {
            y = Permute(y, [0, 2, 3, 1]);
        }

        SetDyn(n, 0, Req(y, ins[3]!, Opt(ins, 4), name));
    }
}
