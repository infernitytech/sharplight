using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Import.Onnx;

// Value-range analysis for runtime tensors whose values define an output extent (Range over a runtime limit). Bounds are
// proven from the ONNX graph: constants, shapes, comparisons (0/1), reductions over known extents, the declared bounds of
// graph inputs (OnnxImportOptions.InputValueBounds) and value-preserving ops. Anything else has no bound.
internal sealed partial class OnnxTranslator
{
    private const int MaxBoundDepth = 32;

    private Dictionary<string, OnnxNode>? _producers;

    private readonly Dictionary<(string Limit, Value Candidates, bool Ascending), Value> _runtimeCounts = [];

    /// <summary>
    /// A proven range <c>[Low, High]</c> of every element of the named tensor (the upper end may be a symbolic extent), or
    /// <see langword="null"/> when the analysis cannot bound it.
    /// </summary>
    private (double Low, Dim High)? ValueBound(string name, int depth = 0)
    {
        if (depth > MaxBoundDepth || name == "")
        {
            return null;
        }

        if (_env.TryGetValue(name, out var known) && known.IsConst)
        {
            return ConstantBound(known.C!);
        }

        if (_opts.InputValueBounds.TryGetValue(name, out long declared))
        {
            return (0, declared);
        }

        if (!Producers().TryGetValue(name, out var n))
        {
            return null;
        }

        (double Low, Dim High)? In(int i) => i < n.Inputs.Count ? ValueBound(n.Inputs[i], depth + 1) : null;
        switch (n.OpType)
        {
            case "Equal" or "Less" or "Greater" or "LessOrEqual" or "GreaterOrEqual" or "Not" or "And" or "Or" or "Xor" or "IsNaN" or "IsInf":
                return (0, 1);
            case "Cast" or "Identity" or "Squeeze" or "Unsqueeze" or "Reshape" or "Flatten" or "Expand" or "Transpose" or "Gather" or "GatherElements"
                or "GatherND" or "Slice" or "ReduceMax" or "ReduceMin" or "Tile" or "Abs" when n.OpType != "Abs" || In(0) is { Low: >= 0 }:
                return In(0);
            case "Shape" or "Size":
                return ShapeBound(n);
            case "ArgMax" or "ArgMin":
                return Extent(n.Inputs[0], (int)n.Int("axis", 0)) is { } extent ? (0, extent - 1) : null;
            case "ReduceSum":
                return In(0) is { Low: >= 0 } summed && ReducedCount(n) is { } count ? (0, summed.High * count) : null;
            case "Concat" or "Max" or "Min" or "Where":
            {
                var parts = Enumerable.Range(n.OpType == "Where" ? 1 : 0, n.Inputs.Count - (n.OpType == "Where" ? 1 : 0)).Select(In).ToList();
                if (parts.Any(p => p is null))
                {
                    return null;
                }

                double low = parts.Min(p => p!.Value.Low);
                Dim high = parts.Select(p => p!.Value.High).Aggregate(Dim.Max);
                return n.OpType == "Min" ? (low, parts.Select(p => p!.Value.High).Aggregate(Dim.Min)) : (low, high);
            }
            case "Add":
                return In(0) is { } a && In(1) is { } b ? (a.Low + b.Low, a.High + b.High) : null;
            case "Mul":
                return In(0) is { Low: >= 0 } x && In(1) is { Low: >= 0 } y ? (0, x.High * y.High) : null;
            case "Sub":
                return In(0) is { } minuend && ConstantInput(n.Inputs[1]) is { } subtrahend
                    ? (minuend.Low - subtrahend.Max(), minuend.High - (long)Math.Floor(subtrahend.Min())) : null;
            default:
                return null;
        }
    }

    /// <summary>The constant's element range; symbolic elements (shape values) contribute their extent.</summary>
    private static (double Low, Dim High)? ConstantBound(OnnxConst c)
    {
        if (c.Count == 0)
        {
            return null;
        }

        if (!c.HasSymbols)
        {
            var values = c.Floats();
            return (values.Min(), (long)Math.Ceiling(values.Max()));
        }

        var ints = c.Ints();
        var fixedValues = ints.Where(v => !SymbolicInts.IsSymbolic(v)).ToList();
        Dim high = ints.Select(v => Dim.FromExpression(SymbolicInts.Decode(v))).Aggregate(Dim.Max);
        return (fixedValues.Count > 0 ? Math.Min(0, fixedValues.Min()) : 0, high);
    }

    private double[]? ConstantInput(string name) => _env.TryGetValue(name, out var t) && t.IsConst && !t.C!.HasSymbols ? t.C!.Floats() : null;

    /// <summary>Elements of a Shape / Size result are bounded by the largest extent (or the element count) of the input.</summary>
    private (double Low, Dim High)? ShapeBound(OnnxNode n)
    {
        if (!_env.TryGetValue(n.Inputs[0], out var t))
        {
            return null;
        }

        var dims = t.IsConst ? t.C!.Dims.Select(d => (Dim)d).ToArray() : t.V!.Shape.Dims.ToArray();
        if (n.OpType == "Size")
        {
            return (0, dims.Aggregate((Dim)1, (a, b) => a * b));
        }

        return dims.Length == 0 ? null : (0, dims.Aggregate(Dim.Max));
    }

    /// <summary>The extent of one axis of a translated tensor (in ONNX rank, i.e. including leading unit dims).</summary>
    private Dim? Extent(string name, int axis)
    {
        if (!_env.TryGetValue(name, out var t))
        {
            return null;
        }

        var dims = OnnxDims(t);
        int a = axis < 0 ? axis + dims.Length : axis;
        return a >= 0 && a < dims.Length ? dims[a] : null;
    }

    private static Dim[] OnnxDims(TranslatedTensor t) =>
        t.IsConst ? t.C!.Dims.Select(d => (Dim)d).ToArray() : [.. Enumerable.Repeat((Dim)1, t.LeadingUnits), .. t.V!.Shape.Dims];

    /// <summary>Number of elements a ReduceSum adds per output element (all of them without axes).</summary>
    private Dim? ReducedCount(OnnxNode n)
    {
        if (!_env.TryGetValue(n.Inputs[0], out var t))
        {
            return null;
        }

        var dims = OnnxDims(t);
        long[]? axes = n.Ints("axes") ?? (n.Inputs.Count > 1 && n.Inputs[1] != "" ? ConstantInput(n.Inputs[1])?.Select(v => (long)v).ToArray() : null);
        if (axes is null || axes.Length == 0)
        {
            return n.Int("noop_with_empty_axes", 0) != 0 ? 1 : dims.Aggregate((Dim)1, (a, b) => a * b);
        }

        return axes.Select(a => dims[a < 0 ? a + dims.Length : a]).Aggregate((Dim)1, (a, b) => a * b);
    }

    /// <summary>
    /// Range with a runtime limit and constant start / delta: the result has a dynamic length bounded by the limit's proven
    /// value range. It is the prefix of the constant sequence <c>start + i · delta</c> (i below the bound) whose elements are
    /// still short of the limit, selected with NonZero.
    /// </summary>
    private void RuntimeRange(OnnxNode n, TranslatedTensor?[] ins)
    {
        var start = ins[0]!.C;
        var delta = ins[2]!.C;
        if (start is null || delta is null || start.HasSymbols || delta.HasSymbols || ins[1]!.IsConst)
        {
            throw new NotSupportedException("Range over runtime tensor values has a data-dependent length (only a runtime limit with constant start and delta is supported)");
        }

        double first = start.Floats()[0];
        double step = delta.Floats()[0];
        if (step == 0)
        {
            throw new NotSupportedException("Range with a zero delta");
        }

        if (ValueBound(n.Inputs[1]) is not { } limitBound)
        {
            throw new NotSupportedException("Range over a runtime limit without a provable upper bound (declare one for graph inputs with OnnxImportOptions.InputValueBounds)");
        }

        // Elements of a positive-step range stay below the limit's upper bound, a negative-step one above its lower bound.
        long count;
        if (step > 0)
        {
            if (limitBound.High.IsSymbolic)
            {
                throw new NotSupportedException("Range over a runtime limit bounded by a symbolic extent");
            }

            count = (long)Math.Max(0, Math.Ceiling((limitBound.High.Fixed - first) / step));
        }
        else
        {
            count = (long)Math.Max(0, Math.Ceiling((limitBound.Low - first) / step));
        }

        if (count == 0)
        {
            throw new NotSupportedException("Range whose proven length is zero (zero-size tensors are not supported)");
        }

        bool integer = !OnnxType.IsFloat(start.ElemType);
        var sequence = new double[count];
        for (long i = 0; i < count; i++)
        {
            sequence[i] = first + i * step;
        }

        string name = Nm(n);
        var candidates = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [count], sequence, name + ".sequence"), name + ".sequence");
        var indices = RuntimeCount(n.Inputs[1], X(ins, 1), candidates, step > 0, name);
        var values = Emit("GatherAxis", [candidates, indices], Attributes.From(("axis", 0L)), integer ? null : name);
        SetDyn(n, 0, integer ? Emit("Cast", [values], Attributes.From(("type", "i32")), name) : values);
    }

    /// <summary>
    /// Indices <c>[count]</c> of the candidate values still short of a runtime limit (below it for an ascending sequence, above it for a
    /// descending one); count is a dynamic symbol. Requests over the same limit tensor (through casts and reshapes) and the same candidates
    /// share one symbol, so that tensors sized by the same runtime value broadcast against each other.
    /// </summary>
    private Value RuntimeCount(string limitName, Value limit, Value candidates, bool ascending, string name)
    {
        var key = (CanonicalSource(limitName), candidates, ascending);
        if (_runtimeCounts.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var limitF32 = limit.Type == DataType.F32 ? limit : Emit("Cast", [limit], Attributes.From(("type", "f32")));
        var inRange = _b.Binary(ascending ? "Less" : "Greater", candidates, limitF32);
        string symbol = NewDynamicSymbol("range", candidates.Shape[0].Fixed);
        return _runtimeCounts[key] = Rs(Emit("NonZero", [inRange], Attributes.From(("count_symbol", symbol)), name + ".indices"), 0);
    }

    /// <summary>The tensor a value is a cast or reshape of (for one-element values such as a runtime extent).</summary>
    private string CanonicalSource(string name)
    {
        var producers = Producers();
        for (int depth = 0; depth < MaxBoundDepth && producers.TryGetValue(name, out var n) && n.OpType is "Cast" or "Identity" or "Squeeze" or "Unsqueeze"; depth++)
        {
            name = n.Inputs[0];
        }

        return name;
    }

    private Dictionary<string, OnnxNode> Producers() =>
        _producers ??= _graph.Nodes.SelectMany(node => node.Outputs.Where(output => output != "").Select(output => (output, node)))
            .GroupBy(pair => pair.output).ToDictionary(group => group.Key, group => group.First().node);

    /// <summary>
    /// ConstantOfShape whose shape has one runtime element (e.g. the largest word count of a batch): that extent becomes a dynamic symbol
    /// bounded by the element's proven value range, and the unit dims in front of it become leading units of the result.
    /// </summary>
    private void RuntimeConstantOfShape(OnnxNode n)
    {
        string shapeName = CanonicalSource(n.Inputs[0]);
        List<string> pieces = Producers().TryGetValue(shapeName, out var concat) && concat.OpType == "Concat" ? [.. concat.Inputs] : [n.Inputs[0]];
        var known = pieces.Select(p => _env.TryGetValue(p, out var t) ? t : null).ToList();
        if (known.Any(t => t is null || (t.IsConst ? t.C!.Count != 1 || t.C.HasSymbols : t.V!.Shape.Elements() != 1)) || known.Count(t => !t!.IsConst) != 1)
        {
            throw new NotSupportedException("ConstantOfShape shape must be a compile-time constant (a runtime shape is supported only when exactly one extent is a runtime value)");
        }

        int axis = known.FindIndex(t => !t!.IsConst);
        long[] dims = [.. known.Select(t => t!.IsConst ? t.C!.Ints()[0] : 0L)];
        if (dims.Take(axis).Any(d => d != 1))
        {
            throw new NotSupportedException("ConstantOfShape shape must be a compile-time constant (a runtime extent is supported only behind unit dims)");
        }

        if (ValueBound(pieces[axis]) is not { High.IsSymbolic: false } bound || bound.High.Fixed <= 0)
        {
            throw new NotSupportedException("ConstantOfShape with a runtime extent without a provable upper bound (declare one for graph inputs with OnnxImportOptions.InputValueBounds)");
        }

        string name = Nm(n);
        long count = bound.High.Fixed;
        var positions = new double[count];
        for (long i = 0; i < count; i++)
        {
            positions[i] = i;
        }

        var candidates = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [count], positions, "positions." + count), "positions." + count);
        var indices = RuntimeCount(pieces[axis], AsValue(known[axis]!), candidates, ascending: true, name);
        var value = n.Tensor("value") is { } t ? OnnxConst.FromTensor(t) : OnnxConst.Scalar(0);
        var type = value.IsFloat ? DataType.F32 : DataType.I32;
        SetDyn(n, 0, FillOuter(SymbolicExpression.Parse(indices.Shape[0].Symbol!), dims[(axis + 1)..], type, value.Floats()[0], 0, name));
        _env[n.Outputs[0]].LeadingUnits = axis;
    }
}
