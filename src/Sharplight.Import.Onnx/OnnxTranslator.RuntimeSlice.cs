using Sharplight.Graph;

namespace Sharplight.Import.Onnx;

// Slice with starts / ends computed at run time (num_logits_to_keep in decoder exports): the kept positions are a data-dependent
// contiguous run of the axis, selected with a mask and NonZero, so the sliced extent is a dynamic symbol bounded by the axis.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Slice of one fixed axis whose starts or ends are runtime tensors (step 1). The axis must be the outermost one that is not a unit
    /// dim, because only the outermost IR dim can be data-dependent; the unit dims in front become leading units of the result.
    /// </summary>
    private void RuntimeSlice(OnnxNode n, TranslatedTensor?[] ins)
    {
        var input = ins[0]!;
        if (!input.IsTensor || input.LeadingUnits > 0 || input.TransposedNonZero)
        {
            throw new NotSupportedException("Slice with runtime starts or ends of a tensor with a data-dependent layout (otherwise starts, ends, axes and steps must be compile-time constants)");
        }

        var x = X(ins, 0);
        long axisCount = ins[1]!.IsConst ? ins[1]!.C!.Count : ins[1]!.V!.Shape.Elements();
        long[] axes = OptConst(ins, 3, "Slice axes")?.Ints() ?? [.. Enumerable.Range(0, (int)axisCount).Select(i => (long)i)];
        long[]? steps = OptConst(ins, 4, "Slice steps")?.Ints();
        if (axes.Length != 1 || steps is not (null or [1]))
        {
            throw new NotSupportedException("Slice with runtime starts or ends is supported on one axis with step 1 (otherwise starts, ends, axes and steps must be compile-time constants)");
        }

        int ax = NormalizeAxis(axes[0], x.Rank);
        if (x.Shape.Dims.Any(d => d.IsSymbolic))
        {
            throw new NotSupportedException("Slice with runtime starts or ends of a tensor with a symbolic dim (otherwise starts, ends, axes and steps must be compile-time constants)");
        }

        if (Enumerable.Range(0, ax).Any(d => x.Shape[d].Fixed != 1))
        {
            throw new NotSupportedException("Slice with runtime starts or ends is supported only along the outermost axis that is not a unit dim (otherwise starts, ends, axes and steps must be compile-time constants)");
        }

        string name = Nm(n);
        long extent = x.Shape[ax].Fixed;
        var positions = new double[extent];
        for (long i = 0; i < extent; i++)
        {
            positions[i] = i;
        }

        var position = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [extent], positions, name + ".positions"), name + ".positions");
        var start = ClampedSliceBound(X(ins, 1), extent, name + ".start");
        var end = ClampedSliceBound(X(ins, 2), extent, name + ".end");
        var kept = _b.Binary("And", _b.Binary("GreaterEqual", position, start), _b.Binary("Less", position, end));
        string symbol = NewDynamicSymbol("slice", extent);
        var indices = Rs(Emit("NonZero", [kept], Attributes.From(("count_symbol", symbol))), 0);
        var rows = ax == 0 ? x : Rs(x, [.. x.Shape.Dims.Skip(ax).Select(d => d.Fixed)]);
        SetDyn(n, 0, Emit("GatherAxis", [rows, indices], Attributes.From(("axis", 0L)), name));
        _env[n.Outputs[0]].LeadingUnits = ax;
    }

    /// <summary>An ONNX Slice bound (f32 [1]) made non-negative (a negative bound counts from the end) and clamped to <c>[0, extent]</c>.</summary>
    private Value ClampedSliceBound(Value bound, long extent, string name)
    {
        var size = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [1], [extent], name + ".extent"), name + ".extent");
        var zero = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [1], [0], name + ".zero"), name + ".zero");
        var fromEnd = _b.Mul(_b.Binary("Less", bound, zero), size);
        return _b.Binary("Max", _b.Binary("Min", _b.Add(bound, fromEnd), size), zero, name);
    }
}
