using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// QuantizeLinear / DequantizeLinear (opset 10..25): per-tensor, per-axis and blocked scales and zero points, constant or runtime, every
// quantized type (8/16-bit integers, 4- and 2-bit integers, FP8 with saturate, FP4), output_dtype and precision. Quantized values travel
// as f32 holding the exact values of the quantized type.
internal sealed partial class OnnxTranslator
{
    private void QuantizeDequantize(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", 1), Math.Max(1, x.Rank));
        long blockSize = n.Int("block_size", 0);
        var scale = QuantParameter(ins[1]!, x, axis, blockSize, name + ".scale");
        var zeroPointTensor = Opt(ins, 2);
        bool zeroPointIsZero = zeroPointTensor is null || (zeroPointTensor.IsConst && zeroPointTensor.C!.Floats().All(z => z == 0));
        var zeroPoint = zeroPointIsZero ? null : QuantParameter(zeroPointTensor!, x, axis, blockSize, name + ".zp");
        int outputDtype = (int)n.Int("output_dtype", 0);
        if (n.OpType == "DequantizeLinear")
        {
            var shifted = zeroPoint is null ? x : _b.Sub(x, zeroPoint);
            int outType = outputDtype != 0 ? outputDtype : ElemTypeOf(ins[1]!);
            var y = Emit("Mul", [shifted, scale], null, OnnxType.NarrowFormat(outType) is null ? name : null);
            if (OnnxType.NarrowFormat(outType) is { } format)
            {
                y = Emit("Narrow", [y], Attributes.From(("format", NumericFormats.Name(format))), name);
            }

            _env[n.Outputs[0]] = Typed(y, outType);
            return;
        }

        int type = outputDtype != 0 ? outputDtype : zeroPointTensor is not null ? ElemTypeOf(zeroPointTensor) : OnnxType.UInt8;
        _env[n.Outputs[0]] = Typed(Quantize(x, scale, zeroPoint, type, n.Int("saturate", 1) != 0, (int)n.Int("precision", 0), name), type);
    }

    /// <summary>y = saturate(round(x / scale) + zero point) for integer types; FP8 / FP4: x / scale (+ zero point) rounded to the type.</summary>
    private Value Quantize(Value x, Value scale, Value? zeroPoint, int type, bool saturate, int precision, string name)
    {
        Value Precise(Value v) => OnnxType.NarrowFormat(precision) is { } p && OnnxType.IsFloat(precision)
            ? Emit("Narrow", [v], Attributes.From(("format", NumericFormats.Name(p))))
            : v;

        var q = Precise(_b.Binary("Div", Precise(x), Precise(scale)));   // a true division (not * 1/s) so ties round like onnxruntime
        var format = OnnxType.NarrowFormat(type) ?? throw new NotSupportedException($"QuantizeLinear to {OnnxType.Name(type)}");
        if (!NumericFormats.IsFloat(format))
        {
            q = Emit("Round", [q]);
            if (zeroPoint is not null)
            {
                q = _b.Add(q, zeroPoint);
            }

            var (lo, hi) = NumericFormats.IntegerRange(format);
            return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
        }

        if (zeroPoint is not null && type == OnnxType.Float4E2M1)
        {
            q = _b.Add(q, zeroPoint);
        }

        return Emit("Narrow", [q], Attributes.From(("format", NumericFormats.Name(format)), ("saturate", saturate)), name);
    }

    /// <summary>
    /// A scale or zero point as an f32 value broadcastable against x: per-tensor (one element), per-axis (a vector along
    /// <paramref name="axis"/>) or blocked (x's shape with ceil(extent / block) entries along the axis, repeated to the full extent).
    /// </summary>
    private Value QuantParameter(TranslatedTensor t, Value x, int axis, long blockSize, string name)
    {
        long count = t.IsConst ? t.C!.Count : t.V!.Shape.Elements();
        if (count == 1)
        {
            return t.IsConst ? ConstValue(t.C!.CastTo(OnnxType.Float).Reshaped([1]), name) : Rs(AsValue(t), 1);
        }

        var dims = Enumerable.Repeat(1L, x.Rank).ToArray();
        if (blockSize <= 0)
        {
            dims[axis] = count;
            return t.IsConst ? ConstValue(t.C!.CastTo(OnnxType.Float).Reshaped(dims), name) : Rs(AsValue(t), dims);
        }

        long extent = x.Shape[axis].Fixed;
        var blockOf = Enumerable.Range(0, (int)extent).Select(i => (long)(i / blockSize)).ToArray();
        if (t.IsConst)
        {
            return ConstValue(RepeatBlocks(t.C!.CastTo(OnnxType.Float), axis, blockOf), name);
        }

        var indices = Idx(TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [extent], blockOf)));
        return Emit("GatherAxis", [AsValue(t), indices], Attributes.From(("axis", (long)axis)), name);
    }

    /// <summary>The constant with entry <c>blockOf[i]</c> of <paramref name="axis"/> at position i (expands blocked parameters).</summary>
    private static OnnxConst RepeatBlocks(OnnxConst c, int axis, long[] blockOf)
    {
        var source = c.Floats();
        var dims = (long[])c.Dims.Clone();
        long blocks = dims[axis];
        dims[axis] = blockOf.Length;
        long inner = 1;
        for (int d = axis + 1; d < dims.Length; d++)
        {
            inner *= dims[d];
        }

        long outer = source.Length / (blocks * inner);
        var values = new double[outer * blockOf.Length * inner];
        for (long o = 0; o < outer; o++)
        {
            for (long i = 0; i < blockOf.Length; i++)
            {
                Array.Copy(source, (o * blocks + blockOf[i]) * inner, values, (o * blockOf.Length + i) * inner, inner);
            }
        }

        return OnnxConst.FromFloats(OnnxType.Float, dims, values, c.Name);
    }
}
