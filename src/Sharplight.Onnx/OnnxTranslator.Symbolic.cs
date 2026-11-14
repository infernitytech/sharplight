using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Shape values that depend on symbolic dimensions (see SymbolicInts): tensors whose extent is a symbolic expression
// (Range / ConstantOfShape / Expand over N) and symbolic shape values used as tensor data.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Whether a node whose inputs are all constants must still be translated into IR because a constant holds a symbolic extent
    /// that becomes a tensor dimension.
    /// </summary>
    private static bool BuildsSymbolicExtent(OnnxNode n, TranslatedTensor?[] ins) => n.OpType switch
    {
        "Expand" => ins[1]!.C!.HasSymbols,
        "Range" => ins.Any(t => t?.C is { HasSymbols: true }),
        "ConstantOfShape" => ins[0]!.C!.HasSymbols,
        "Tile" => ins[1]!.C!.HasSymbols,
        _ => false,
    };

    /// <summary><c>y[i, ...] = start + i · delta</c> with outer extent <paramref name="outer"/> and fixed inner dims.</summary>
    private Value FillOuter(SymbolicExpression outer, long[] inner, DataType type, double start, double delta, string? name = null)
    {
        var attributes = Attributes.From(("shape", (long[])[0, .. inner]), ("outer_symbol", outer.ToString()), ("start", start), ("delta", delta), ("type", type.Name));
        return Emit("Fill", [], attributes, name);
    }

    /// <summary>ONNX Range whose limit (or start, with a constant length) is symbolic.</summary>
    private void Range(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (ins.Any(t => t is null || !t.IsConst))
        {
            RuntimeRange(n, ins);
            return;
        }

        var start = ins[0]!.C!;
        var limit = ins[1]!.C!;
        var delta = ins[2]!.C!;
        if (start.HasSymbols || delta.HasSymbols)
        {
            throw new NotSupportedException("Range with a symbolic start or step");
        }

        long step = delta.Ints()[0];
        if (start.IsFloat || delta.IsFloat || step == 0)
        {
            throw new NotSupportedException("floating point Range over a symbolic limit");
        }

        long first = start.Ints()[0];
        var span = SymbolicInts.Decode(limit.Ints()[0]) - first;
        var count = step > 0 ? span.CeilDiv(step) : (-span).CeilDiv(-step);
        var type = OnnxType.IsFloat(start.ElemType) ? DataType.F32 : DataType.I32;
        SetDyn(n, 0, FillOuter(NonNegative(count), [], type, first, step, Nm(n)));
    }

    /// <summary>ONNX ConstantOfShape whose shape has a symbolic outermost extent.</summary>
    private void ConstantOfShape(OnnxNode n, TranslatedTensor?[] ins)
    {
        var dims = ConstIn(ins, 0, "ConstantOfShape shape", symbolic: true).Ints();
        if (dims.Length == 0 || dims.Skip(1).Any(SymbolicInts.IsSymbolic))
        {
            throw new NotSupportedException($"ConstantOfShape [{string.Join(", ", dims.Select(SymbolicInts.Format))}] has a symbolic inner extent");
        }

        var value = n.Tensor("value") is { } t ? OnnxConst.FromTensor(t) : OnnxConst.Scalar(0);
        var type = value.IsFloat ? DataType.F32 : DataType.I32;
        SetDyn(n, 0, FillOuter(SymbolicInts.Decode(dims[0]), dims[1..], type, value.Floats()[0], 0, Nm(n)));
    }

    /// <summary>
    /// A constant with symbolic elements used as tensor data (e.g. a sequence length that scales attention scores): each symbolic element
    /// is computed at run time as the sum of a <see cref="StandardOps.Fill"/> of ones with that extent.
    /// </summary>
    private Value SymbolicData(OnnxConst c, DataType type, string? hint)
    {
        var values = c.Floats();   // symbolic elements are NaN-coded (SymbolicInts.ToDouble)
        var parts = new Value[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            if (SymbolicInts.IsSymbolicDouble(values[i]))
            {
                var ones = FillOuter(SymbolicInts.Decode(SymbolicInts.FromDouble(values[i])), [], DataType.F32, 1, 0);
                parts[i] = Emit("Reduce", [ones], Attributes.From(("mode", "sum"), ("axes", L(0)), ("keepdims", true)));
            }
            else
            {
                parts[i] = _b.Constant($"{hint ?? "shape"}.{i}", DataType.F32, new Shape(1), BitConverter.GetBytes((float)values[i]));
            }
        }

        var flat = parts.Length == 1 ? parts[0] : ConcatMany(0, parts);
        var shaped = c.Rank <= 1 ? flat : Rs(flat, c.Dims);
        return type == DataType.F32 ? shaped : Emit("Cast", [shaped], Attributes.From(("type", type.Name)));
    }

    /// <summary><paramref name="e"/>, or <c>max(0, e)</c> unless it is provably non-negative for symbols ≥ 1.</summary>
    private static SymbolicExpression NonNegative(SymbolicExpression e)
    {
        try
        {
            if (SymbolicInts.CompareSigned(SymbolicInts.Encode(e), -1) > 0)
            {
                return e;
            }
        }
        catch (NotSupportedException)
        {
            // Undecidable: keep the clamp.
        }

        return SymbolicExpression.Max(SymbolicExpression.Constant(0), e);
    }
}
