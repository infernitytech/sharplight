using Sharplight.Graph;

namespace Sharplight.Onnx;

// com.microsoft quantized linear operators of the onnxruntime quantization tools (integer accumulation carried exactly in f32).
internal sealed partial class OnnxTranslator
{
    /// <summary>(a - a_zp) · (B - b_zp) as an f32 product of exact integers; B constant (per-column zero points) or runtime.</summary>
    private Value IntegerProduct(Value a, TranslatedTensor b, TranslatedTensor? bZeroPoint, bool transB, string name)
    {
        if (b.IsConst && bZeroPoint is null or { IsConst: true })
        {
            var bc = b.C!;
            var shifted = DequantConst(bc, OnnxConst.Scalar(1f), bZeroPoint?.C, transB ? 0 : bc.Rank - 1);
            return MatMulWeightOrBatch(a, transB ? shifted.Transposed([1, 0]) : shifted, name);
        }

        var bv = bZeroPoint is null ? AsValue(b) : _b.Sub(AsValue(b), QParam(bZeroPoint, name + ".bzp"));
        var y = Emit("BatchMatMul", [a, bv], Attributes.From(("trans_b", transB)), name);
        Heavy();
        return y;
    }

    /// <summary>MatMulIntegerToFloat: (A - a_zp) · (B - b_zp) · a_scale · b_scale (+ bias).</summary>
    private void MatMulIntegerToFloat(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var acc = IntegerProduct(ShiftZp(ins[0]!, Opt(ins, 4), name), ins[1]!, Opt(ins, 5), false, name + ".acc");
        var y = _b.Mul(acc, _b.Mul(QParam(ins[2]!, name + ".as"), QParam(ins[3]!, name + ".bs")));
        if (Opt(ins, 6) is not null)
        {
            y = _b.Add(y, X(ins, 6));
        }

        SetDyn(n, 0, y);
    }

    /// <summary>DynamicQuantizeMatMul: A quantized like DynamicQuantizeLinear (uint8), then MatMulIntegerToFloat with B's scale and zero point.</summary>
    private void DynamicQuantizeMatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var (q, scale, zeroPoint) = DynamicQuantizeValues(X(ins, 0), name + ".a");
        var acc = IntegerProduct(_b.Sub(q, zeroPoint), ins[1]!, Opt(ins, 3), false, name + ".acc");
        var y = _b.Mul(acc, _b.Mul(scale, QParam(ins[2]!, name + ".bs")));
        if (Opt(ins, 4) is not null)
        {
            y = _b.Add(y, X(ins, 4));
        }

        SetDyn(n, 0, y);
    }

    /// <summary>
    /// QGemm: alpha · a_scale · b_scale · ((A - a_zp)' · (B - b_zp)' + C) with C an int32 bias of that scale; float output, or quantized
    /// with y_scale / y_zero_point.
    /// </summary>
    private void QGemm(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var a = ShiftZp(ins[0]!, Opt(ins, 2), name);
        if (n.Int("transA", 0) != 0)
        {
            a = Permute(a, [1, 0]);
        }

        var acc = IntegerProduct(a, ins[3]!, Opt(ins, 5), n.Int("transB", 0) != 0, name + ".acc");
        if (Opt(ins, 6) is not null)
        {
            acc = _b.Add(acc, X(ins, 6));
        }

        var y = _b.Mul(acc, _b.Scale(_b.Mul(QParam(ins[1]!, name + ".as"), QParam(ins[4]!, name + ".bs")), n.Float("alpha", 1f)));
        if (Opt(ins, 7) is not { } yScale)
        {
            SetDyn(n, 0, y);
            return;
        }

        var yZeroPoint = Opt(ins, 8);
        var quantized = Emit("Round", [_b.Binary("Div", y, QParam(yScale, name + ".ys"))]);
        if (yZeroPoint is not null)
        {
            quantized = _b.Add(quantized, QParam(yZeroPoint, name + ".yzp"));
        }

        int type = yZeroPoint is not null ? ElemTypeOf(yZeroPoint) : OnnxType.UInt8;
        var (lo, hi) = OnnxFold.Range(type);
        _env[n.Outputs[0]] = Typed(Emit("Clip", [quantized], Attributes.From(("min", (float)lo), ("max", (float)hi)), name), type);
    }

    /// <summary>MatMulInteger16: int16 · int16 → int32, exact in f32 while the sums stay below 2^24.</summary>
    private void MatMulInteger16(OnnxNode n, TranslatedTensor?[] ins)
    {
        var y = Emit("BatchMatMul", [X(ins, 0), X(ins, 1)], null, Nm(n));
        Heavy();
        _env[n.Outputs[0]] = Typed(y, OnnxType.Int32);
    }

    /// <summary>QLinearWhere: Where(condition, dequantize(X), dequantize(Y)) requantized with z_scale / z_zero_point.</summary>
    private void QLinearWhere(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        Value Dequantize(int value, int scale, int zeroPoint, string suffix)
            => _b.Mul(_b.Sub(AsValue(ins[value]!), QParam(ins[zeroPoint]!, name + suffix + "zp")), QParam(ins[scale]!, name + suffix + "s"));
        var selected = Emit("Where", [AsValue(ins[0]!), Dequantize(1, 2, 3, ".x"), Dequantize(4, 5, 6, ".y")]);
        var quantized = _b.Add(Emit("Round", [_b.Binary("Div", selected, QParam(ins[7]!, name + ".zs"))]), QParam(ins[8]!, name + ".zzp"));
        int type = ElemTypeOf(ins[8]!);
        var (lo, hi) = OnnxFold.Range(type);
        _env[n.Outputs[0]] = Typed(Emit("Clip", [quantized], Attributes.From(("min", (float)lo), ("max", (float)hi)), name), type);
    }
}
