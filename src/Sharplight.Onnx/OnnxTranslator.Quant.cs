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
    private Value QParam(TranslatedTensor t, string name) => t.IsConst ? ConstValue(t.C!.CastTo(OnnxType.Float).Count == 1 ? t.C.CastTo(OnnxType.Float).Reshaped([1]) : t.C.CastTo(OnnxType.Float), name) : t.V!;

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

    private void QuantizeDequantize(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var sc = ConstIn(ins, 1, "scale");
        var zp = OptConst(ins, 2, "zero point");
        int axis = (int)n.Int("axis", 1);
        var scV = ConstValue(Bcast(sc.CastTo(OnnxType.Float), x.Rank, axis), Nm(n) + ".scale");
        Value zpV = zp is null ? null! : ConstValue(Bcast(zp.CastTo(OnnxType.Float), x.Rank, axis), Nm(n) + ".zp");
        if (n.OpType == "DequantizeLinear")
        {
            var v = zp is null ? x : _b.Sub(x, zpV);
            SetDyn(n, 0, Emit("Mul", [v, scV], null, Nm(n)));
            return;
        }
        int et = zp?.ElemType ?? OnnxType.UInt8;
        var q = Emit("Round", [_b.Binary("Div", x, scV)]);
        if (zp is not null)
        {
            q = _b.Add(q, zpV);
        }

        var (lo, hi) = OnnxFold.Range(et);
        SetDyn(n, 0, Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), Nm(n)));
    }

    // ---- quantization ----------------------------------------------------------------------------------------
    /// <summary>Per-axis constant (scale/zero point) shaped for broadcasting against a tensor of rank <paramref name="rank"/>.</summary>
    private OnnxConst Bcast(OnnxConst c, int rank, int axis)
    {
        if (c.Count == 1)
        {
            return c.Reshaped([1]);
        }

        var d = Enumerable.Repeat(1L, rank).ToArray();
        d[axis < 0 ? axis + rank : axis] = c.Count;
        return c.Reshaped(d);
    }

    /// <summary>Dequantizes a constant (per-tensor/per-axis zero point and scale) to f32.</summary>
    private static OnnxConst DequantConst(OnnxConst x, OnnxConst scale, OnnxConst? zp, int axis = 0) => OnnxFold.Dequantize(x, scale, zp, axis);

    private Value ShiftZp(TranslatedTensor a, TranslatedTensor? zpT, string nm)
    {
        if (zpT is { IsConst: false })
        {
            return _b.Sub(AsValue(a), zpT.V!);   // runtime zero point (DynamicQuantizeLinear)
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

    private Value Requant(Value y, float yScale, OnnxConst? yzp, string name)
    {
        var q = Emit("Round", [_b.Scale(y, 1f / yScale)]);
        int et = yzp?.ElemType ?? OnnxType.UInt8;
        if (yzp is not null)
        {
            q = _b.Scale(q, 1f, yzp.ScalarF32());
        }

        var (lo, hi) = OnnxFold.Range(et);
        return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
    }

    /// <summary>
    /// y = clip(round(x / s) + zp, 0, 255) with s = (max(x,0) - min(x,0)) / 255 and zp = clip(round(-min(x,0) / s), 0, 255), over the whole tensor.
    /// </summary>
    private void DynamicQuantize(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var all = Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray();
        var maxAttributes = Attributes.From(("mode", "max"), ("axes", all), ("keepdims", true));
        var minAttributes = Attributes.From(("mode", "min"), ("axes", all), ("keepdims", true));
        var hi = Emit("Clip", [Emit("Reduce", [x], maxAttributes)], Attributes.From(("min", 0f)));
        var lo = Emit("Clip", [Emit("Reduce", [x], minAttributes)], Attributes.From(("max", 0f)));
        var scale = _b.Scale(_b.Sub(hi, lo), 1f / 255f);
        var byteRange = Attributes.From(("min", 0f), ("max", 255f));
        var zeroPoint = Emit("Clip", [Emit("Round", [_b.Binary("Div", _b.Scale(lo, -1f), scale)])], byteRange);
        var y = Emit("Clip", [_b.Add(Emit("Round", [_b.Binary("Div", x, scale)]), zeroPoint)], byteRange, Nm(n));
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

    private void MatMulInteger(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var a = ShiftZp(ins[0]!, Opt(ins, 2), name);
        var bConst = ConstIn(ins, 1, "MatMulInteger B");
        var bZeroPoint = OptConst(ins, 3, "b zero point");
        var b = DequantConst(bConst, OnnxConst.Scalar(1f), bZeroPoint, bConst.Rank - 1);   // per-column zero points when bZeroPoint is a vector
        SetDyn(n, 0, MatMulWeightOrBatch(a, b, name));
    }

    private void QLinearMatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var aScale = ConstIn(ins, 1, "a_scale").ScalarF32();
        var a = _b.Scale(ShiftZp(ins[0]!, OptConst(ins, 2, "a_zp"), name), aScale);
        var b = DequantConst(ConstIn(ins, 3, "B"), ConstIn(ins, 4, "b_scale"), OptConst(ins, 5, "b_zp"), 1);
        var y = MatMulWeightOrBatch(a, b, name + ".mm");
        SetDyn(n, 0, Requant(y, ConstIn(ins, 6, "y_scale").ScalarF32(), OptConst(ins, 7, "y_zp"), name));
    }

    /// <summary>ConvInteger and QLinearConv: dequantize to f32, run a regular Conv and (for QLinearConv) requantize.</summary>
    private void QuantizedConv(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        bool linear = n.OpType == "QLinearConv";
        float inputScale = linear ? ConstIn(ins, 1, "x_scale").ScalarF32() : 1f;
        Value x;
        if (linear)
        {
            x = _b.Scale(ShiftZp(ins[0]!, OptConst(ins, 2, "x_zp"), name), inputScale);
        }
        else
        {
            x = ShiftZp(ins[0]!, Opt(ins, 2), name);
        }

        var weightConst = ConstIn(ins, linear ? 3 : 1, "W");
        var weightScale = linear ? ConstIn(ins, 4, "w_scale") : OnnxConst.Scalar(1f);
        var weightZeroPoint = OptConst(ins, linear ? 5 : 3, "w_zp");
        var weight = DequantConst(weightConst, weightScale, weightZeroPoint, 0);
        OnnxConst? bias = null;
        if (linear && ins.Length > 8 && ins[8] is not null)
        {
            var scales = weightScale.Floats();
            var quantizedBias = ConstIn(ins, 8, "B").Floats();
            var values = quantizedBias.Select((v, i) => v * inputScale * scales[weightScale.Count == 1 ? 0 : i]).ToArray();
            bias = OnnxConst.FromFloats(OnnxType.Float, [weight.Dims[0]], values);
        }

        var convInputs = new TranslatedTensor?[] { TranslatedTensor.Dyn(x), TranslatedTensor.Const(weight.Reshaped(weight.Dims, name + ".w")), bias is null ? null : TranslatedTensor.Const(bias) };
        var conv = new OnnxNode { OpType = "Conv", Name = n.Name, Inputs = ["x", "w", bias is null ? "" : "b"], Outputs = [name + ".conv"], Attributes = n.Attributes };
        Conv(conv, convInputs);
        var y = _env[name + ".conv"].V!;
        SetDyn(n, 0, linear ? Requant(y, ConstIn(ins, 6, "y_scale").ScalarF32(), OptConst(ins, 7, "y_zp"), name) : y);
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
