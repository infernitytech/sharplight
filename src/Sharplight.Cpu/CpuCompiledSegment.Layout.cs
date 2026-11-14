using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

// Channels-last layout planning: which conv / pool / elementwise outputs are stored channels-last (see ChannelsLast).
// A conv writes channels-last when its input already is, or when every reader of its output (through elementwise ops)
// can take that layout; other readers get a channels-last value as a strided view and materialize it, so conversions
// only happen at the borders of channels-last regions. DeviceFeatures.ChannelsLast turns the layout off.
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>Most elementwise ops followed from a value when deciding whether its readers take channels-last input.</summary>
    private const int MaxLayoutDepth = 8;

    private readonly Dictionary<Value, bool> _channelsLastReaders = [];
    private readonly Dictionary<TRef, TRef> _channelsLastCopies = [];

    private bool ChannelsLastEnabled => _features.Contains(DeviceFeatures.ChannelsLast);

    /// <summary>A new channels-last buffer for rank-4 f32 <paramref name="v"/>.</summary>
    private TRef NewChannelsLast(Value v)
    {
        var dims = DimsOf(v.Shape);
        return Register(new TRef
        {
            Type = v.Type,
            Dims = dims,
            Strides = ChannelsLast.Strides(dims),
            Sym = v.Shape.Symbol,
            Buf = NewArenaBuf(ElementCount(dims) * v.Type.BlockBytes, v.Name),
        });
    }

    /// <summary><paramref name="x"/> (rank-4 f32) as a channels-last view: itself, or a channels-last copy made once per source.</summary>
    private TRef ToChannelsLast(TRef x)
    {
        if (ChannelsLast.IsView(x))
        {
            return x;
        }

        if (_channelsLastCopies.TryGetValue(x, out var copy))
        {
            return copy;
        }

        var dims = (long[])x.Dims.Clone();
        var y = Register(new TRef
        {
            Type = x.Type,
            Dims = dims,
            Strides = ChannelsLast.Strides(dims),
            Sym = x.Sym,
            Buf = NewArenaBuf(ElementCount(dims) * x.Type.BlockBytes, "channels_last"),
        });

        // As [N, H, W, C]: the destination is dense, the source a transposed view (the blocked transpose of CopyStrided).
        long[] nhwc = [dims[0], dims[2], dims[3], dims[1]];
        var src = View(x, nhwc, [x.Strides[0], x.Strides[2], x.Strides[3], x.Strides[1]], 0, x.Sym);
        var dst = View(y, nhwc, TRef.RowMajor(nhwc), 0, x.Sym);
        AddStep(new CopyOp(src, dst), x, y);
        _channelsLastCopies[x] = y;
        return y;
    }

    /// <summary>A rank-2 [N, C·H·W] view over the whole buffer of channels-last <paramref name="r"/>, for elementwise ops.</summary>
    private TRef FlatView(TRef r)
    {
        long row = r.Dims[1] * r.Dims[3];
        return View(r, [r.Dims[0], r.Dims[2], row], [r.Dims[2] * row, row, 1], 0, r.Sym);
    }

    /// <summary>The conv kind node <paramref name="n"/> runs as in the channels-last layout: 0 none, 1 dense (groups 1), 2 depthwise.</summary>
    private int ChannelsLastConvKind(Node n)
    {
        var ins = n.Inputs;
        if (!ChannelsLastEnabled || ins[0].Type != Tensors.DataType.F32 || ins[0].Shape.Rank != 4 || ins[1].Shape.Rank != 4 || !IsStatic(ins[1])
            || (ins.Length > 2 && !IsStatic(ins[2])) || n.Output.Shape.Rank != 4)
        {
            return 0;
        }

        for (int i = 1; i < 4; i++)
        {
            if (ins[0].Shape[i].IsSymbolic || n.Output.Shape[i].IsSymbolic)
            {
                return 0;
            }
        }

        long groups = n.Attributes.Get("groups", 1L);
        long c = ins[0].Shape[1].Fixed;
        long o = n.Output.Shape[1].Fixed;
        if (groups == 1)
        {
            return 1;
        }

        return groups == c && o == c && ins[1].Shape[1].Fixed == 1 ? 2 : 0;
    }

    /// <summary>
    /// True when every reader of <paramref name="v"/> takes channels-last input: convs, pooling, spatial reductions, or (looking
    /// through them) elementwise ops of its shape, channel slices and channel concats.
    /// </summary>
    private bool ReadersTakeChannelsLast(Value v, int depth = 0)
    {
        if (_channelsLastReaders.TryGetValue(v, out var known))
        {
            return known;
        }

        List<Node>? readers = null;
        bool ok = depth < MaxLayoutDepth && !_segment.Outputs.Contains(v) && _consumers.TryGetValue(v, out readers) && readers.Count > 0;
        if (ok)
        {
            foreach (var reader in readers!)
            {
                if (!ReaderTakesChannelsLast(reader, v, depth))
                {
                    ok = false;
                    break;
                }
            }
        }

        _channelsLastReaders[v] = ok;
        return ok;
    }

    private bool ReaderTakesChannelsLast(Node reader, Value v, int depth)
    {
        string op = reader.Op.Name;
        switch (op)
        {
            case "Conv":
                return reader.Inputs[0] == v && reader.Inputs.Count(i => i == v) == 1 && ChannelsLastConvKind(reader) != 0;
            case "Pool":
                return reader.Output.Shape.Rank == 4;
            case "Reduce":
                return GlobalSpatialReduce(reader) is not null;
            case "Resize":
                return reader.Inputs[0] == v && reader.Output.Shape.Rank == 4 && ReadersTakeChannelsLast(reader.Output, depth + 1);
            case "Slice":
                return reader.Output.Shape.Rank == 4 && ReadersTakeChannelsLast(reader.Output, depth + 1);
            case "Concat":
                return reader.Output.Shape.Rank == 4 && NormalizeAxis(reader.Attributes.Get<int>("axis"), 4) == 1
                    && ReadersTakeChannelsLast(reader.Output, depth + 1);
        }

        bool elementwise = op == "Clip" || (s_unaryKinds.ContainsKey(op) && !CpuOpSupport.IsIntegerElementwise(reader))
            || (s_binaryKinds.ContainsKey(op) && !CpuOpSupport.IsIntegerElementwise(reader));
        if (!elementwise || !reader.Output.Shape.Equals(v.Shape))
        {
            return false;
        }

        foreach (var input in reader.Inputs)
        {
            bool broadcast = IsStatic(input) && input.Shape.Rank <= 4 && input.Shape.Dims.All(d => !d.IsSymbolic)
                && input.Shape.Dims.Aggregate(1L, (a, d) => a * d.Fixed) <= (v.Shape[1].IsSymbolic ? 0 : v.Shape[1].Fixed);
            bool perImageChannel = input.Shape.Rank == 4 && input.Shape[2].Equals(Tensors.Dim.Of(1)) && input.Shape[3].Equals(Tensors.Dim.Of(1))
                && (input.Shape[0].Equals(Tensors.Dim.Of(1)) || input.Shape[0].Equals(v.Shape[0]))
                && (input.Shape[1].Equals(Tensors.Dim.Of(1)) || input.Shape[1].Equals(v.Shape[1]));
            if (!input.Shape.Equals(v.Shape) && !broadcast && !perImageChannel)
            {
                return false;
            }
        }

        return ReadersTakeChannelsLast(reader.Output, depth + 1);
    }

    /// <summary>The kind of a Reduce over exactly the spatial axes 2 and 3 of a rank-4 input (mean or max), else null.</summary>
    private static ReduceKind? GlobalSpatialReduce(Node n)
    {
        if (n.Inputs[0].Shape.Rank != 4 || n.Inputs[0].Type != Tensors.DataType.F32)
        {
            return null;
        }

        var axes = n.Attributes.Get<long[]>("axes").Select(a => a < 0 ? a + 4 : a).Order().ToArray();
        if (axes is not [2, 3])
        {
            return null;
        }

        return n.Attributes.Get("mode", "sum") switch
        {
            "mean" => ReduceKind.Mean,
            "max" => ReduceKind.Max,
            _ => null,
        };
    }

    /// <summary>Lowers a conv in the channels-last layout when <see cref="ChannelsLastConvKind"/> allows it and the layout pays; false otherwise.</summary>
    private bool TryLowerChannelsLastConv(Node n)
    {
        int kind = ChannelsLastConvKind(n);
        if (kind == 0)
        {
            return false;
        }

        var ins = n.Inputs;
        var x = RefF32(ins[0]);
        if (!ChannelsLast.IsView(x) && !ReadersTakeChannelsLast(n.Output))
        {
            return false;
        }

        var a = n.Attributes;
        var weight = RefF32(ins[1]);
        TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
        var strides = Ints(a, "strides", 1, 1);
        var pads = Ints(a, "pads", 0, 0, 0, 0);
        var dil = Ints(a, "dilations", 1, 1);
        float* w = (float*)(weight.Buf.Ptr + weight.Offset * 4);
        if (kind == 1)
        {
            x = ChannelsLastConvOp.UsesRowRuns((int)x.Dims[1], dil[1]) && !ChannelsLast.IsView(x) ? Dense(x) : ToChannelsLast(x);
        }
        else
        {
            x = ToChannelsLast(x);
        }

        var y = NewChannelsLast(n.Output);
        var epilogue = AbsorbEpilogue(n.Output, y);
        CpuOp op = kind == 1
            ? ChannelsLastConvOp.Create(x, w, (int)weight.Dims[2], (int)weight.Dims[3], bias, y, strides, pads, dil, epilogue)
            : new ChannelsLastDepthwiseOp(x, w, (int)weight.Dims[2], (int)weight.Dims[3], bias, y, strides, pads, dil, epilogue);
        AddStep(op, Touched(x, bias, y, epilogue?.Residual));
        _refs[n.Output] = y;
        return true;
    }

    /// <summary>Pooling of a channels-last input stays channels-last; false when the input is not channels-last.</summary>
    private bool TryLowerChannelsLastPool(Node n)
    {
        var x = RefF32(n.Inputs[0]);
        if (!ChannelsLastEnabled || !ChannelsLast.IsView(x) || n.Output.Shape.Rank != 4)
        {
            return false;
        }

        var a = n.Attributes;
        var y = NewChannelsLast(n.Output);
        var op = new ChannelsLastPoolOp(
            x, y, a.Get("mode", "max") == "max", Ints(a, "kernel"), Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0),
            Ints(a, "dilations", 1, 1), a.Get("count_include_pad", false));
        AddStep(op, x, y);
        _refs[n.Output] = y;
        return true;
    }

    /// <summary>A mean / max over the spatial axes of a channels-last input; false otherwise.</summary>
    private bool TryLowerChannelsLastReduce(Node n)
    {
        var x = RefF32(n.Inputs[0]);
        if (!ChannelsLastEnabled || !ChannelsLast.IsView(x) || GlobalSpatialReduce(n) is not { } kind)
        {
            return false;
        }

        var y = NewDense(n.Output);
        AddStep(ChannelsLastPoolOp.Global(x, y, kind == ReduceKind.Max), x, y);
        _refs[n.Output] = y;
        return true;
    }

    /// <summary>
    /// Elementwise ops with a rank-4 output whose full-size inputs are all channels-last views (other inputs broadcast, e.g.
    /// per-channel constants) write a channels-last output. They run over [N, H, W, C] views of every operand, or over the
    /// flat buffers when all operands are whole channels-last tensors. False otherwise.
    /// </summary>
    private bool TryChannelsLastElementwise(Node n, TRef[] inputs, out TRef y, out TRef[] views, out TRef yView)
    {
        y = null!;
        views = [];
        yView = null!;
        if (!ChannelsLastEnabled || n.Output.Shape.Rank != 4 || n.Output.Type != Tensors.DataType.F32)
        {
            return false;
        }

        var dims = DimsOf(n.Output.Shape);
        bool any = false;
        bool flat = true;
        foreach (var input in inputs)
        {
            if (input.Rank == 4 && input.Dims.AsSpan().SequenceEqual(dims) && input.Sym == n.Output.Shape.Symbol)
            {
                // A full-size operand in another layout is gathered in place (one transpose either way).
                any |= ChannelsLast.IsView(input);
                flat &= ChannelsLast.Is(input);
            }
            else if (input.Rank <= 4 && input.Sym is null && ElementCount(input.Dims) <= dims[1])
            {
                flat = false;
            }
            else if (input.Rank == 4 && input.Dims[2] == 1 && input.Dims[3] == 1 && (input.Dims[1] == 1 || input.Dims[1] == dims[1])
                && ((input.Dims[0] == 1 && input.Sym is null) || (input.Dims[0] == dims[0] && input.Sym == n.Output.Shape.Symbol)))
            {
                // One value per image and channel (a squeeze-excitation scale).
                flat = false;
            }
            else
            {
                return false;
            }
        }

        if (!any)
        {
            return false;
        }

        y = NewChannelsLast(n.Output);
        views = flat ? inputs.Select(FlatView).ToArray() : inputs.Select(NhwcView).ToArray();
        yView = flat ? FlatView(y) : NhwcView(y);
        return true;
    }

    /// <summary>[N, H, W, C] view of a rank-4 (or lower, broadcast, left-padded with unit dims) value.</summary>
    private TRef NhwcView(TRef r)
    {
        int pad = 4 - r.Rank;
        var dims = new long[4];
        var strides = new long[4];
        for (int i = 0; i < 4; i++)
        {
            dims[i] = i < pad ? 1 : r.Dims[i - pad];
            strides[i] = i < pad ? 0 : r.Strides[i - pad];
        }

        return View(r, [dims[0], dims[2], dims[3], dims[1]], [strides[0], strides[2], strides[3], strides[1]], 0, pad == 0 ? r.Sym : null);
    }

    /// <summary>A Concat along the channels of channels-last views writes a channels-last output; false otherwise.</summary>
    private bool TryLowerChannelsLastConcat(Node n, int axis)
    {
        if (!ChannelsLastEnabled || axis != 1 || n.Output.Shape.Rank != 4)
        {
            return false;
        }

        var xs = n.Inputs.Select(Ref).ToArray();
        if (!xs.All(ChannelsLast.IsView))
        {
            return false;
        }

        var y = NewChannelsLast(n.Output);
        AddStep(new ConcatOp(xs, y, axis), [.. xs, y]);
        _refs[n.Output] = y;
        return true;
    }
}
