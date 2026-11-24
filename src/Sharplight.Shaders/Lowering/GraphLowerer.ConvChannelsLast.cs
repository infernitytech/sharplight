using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Reduced-precision convolutions over channels-last 16-bit activations (conv_cl.slang): f16 where the device has f16 matrices,
/// else bf16 where it has bf16 ones, f32 accumulation. The activations are staged once into the channels-last layout
/// (conv_cl_stage), so the implicit GEMM loads 8 channels of an input pixel with one 16-byte load and one bounds check instead
/// of converting and addressing every value (conv_cm); the constant weights are converted and reordered tap-major on the host.
/// A channel concatenation read only by such a convolution (the stacked frame windows of a causal 3-D convolution, the skip
/// connections of a UNet) is not materialized: each part is staged straight into its channel range of the convolution's input
/// (<see cref="PlanChannelsLastConcats"/>). Strict precision keeps conv_cm's f32-equivalent forms.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>
    /// The conv_cl configuration: the 16-bit type of the operands and the tile (subgroup rows and columns of 64 output channels x
    /// <see cref="FragN"/> 16-pixel fragments each).
    /// </summary>
    private readonly record struct ConvChannelsLastForm(DataType Type, int WarpsM, int WarpsN, int FragN, int Stages)
    {
        /// <summary>Shared operand buffers: the asynchronous copies' ring, else two.</summary>
        public int Buffers => Stages > 0 ? Stages : 2;

        public long TileRows => WarpsM * ConvMatrixWarpRows;

        public long TileColumns => WarpsN * FragN * 16L;

        public int Threads => WarpsM * WarpsN * 32;
    }

    /// <summary>Concatenations lowered as channels-last 16-bit stagings of their parts (see <see cref="PlanChannelsLastConcats"/>).</summary>
    private readonly Dictionary<Value, DataType> _channelsLastConcats = [];

    /// <summary>Parts of those concatenations (or of their parts) that the unit producing them stages in place, by value.</summary>
    private readonly Dictionary<Value, ChannelsLastSlot> _channelsLastParts = [];

    /// <summary>Where a part is staged: the channels-last concatenation output, its type, and the part's channel and batch offsets.</summary>
    private readonly record struct ChannelsLastSlot(Value Concat, DataType Type, long Channel, long Batch);

    /// <summary>The channels-last 16-bit weights [O, KH·KW·CG] (tap-major) of each constant conv weight, by weight and type.</summary>
    private readonly Dictionary<(Value Weight, DataType Type), TensorRef> _channelsLastWeights = [];

    /// <summary>Shared memory of conv_cl: the buffers of both operand tiles (the epilogue reuses the A tiles).</summary>
    private static long ConvChannelsLastSharedBytes(ConvChannelsLastForm form) => form.Buffers * (form.TileRows + form.TileColumns) * (ConvMatrixTileK + 8) * 2;

    /// <summary>True when the A tiles hold the epilogue's row of f32 results of every subgroup (16 x FragN * 16 values each).</summary>
    private static bool ConvChannelsLastEpilogueFits(ConvChannelsLastForm form)
        => form.Buffers * form.TileRows * (ConvMatrixTileK + 8) * 2 >= form.WarpsM * form.WarpsN * form.FragN * 1024L;

    /// <summary>
    /// True when conv_cl streams its tiles with asynchronous copies (<c>CL_ASYNC</c>): a CUDA C++ target on a device of compute
    /// capability 8.0 or later (<see cref="DeviceFeatures.NativeGemmF16"/>, which needs the same instructions).
    /// </summary>
    private bool ConvAsyncCopies => _opt.Has(DeviceFeatures.NativeGemmF16) && _compiler.CompilesCuda(_opt.Target) && DiagnosticSwitches.ConvAsyncCopies;

    /// <summary>
    /// The conv_cl form of convolution <paramref name="n"/>, or null when it does not apply: reduced precision, 16-bit matrices
    /// with 32-lane subgroups, a constant float weight, channel groups of whole 8-value items, at least
    /// <see cref="ConvMatrixMinChannels"/> outputs per group and <see cref="ConvMatrixMinColumns"/> output pixels; not depthwise
    /// and not narrow (those have their own kernels). Decided on the graph alone, so the concatenation planning and the
    /// lowering agree.
    /// </summary>
    private ConvChannelsLastForm? ChooseConvChannelsLast(Node n)
    {
        if (!_opt.Compile.AllowReducedPrecision || !DiagnosticSwitches.ConvChannelsLast || DiagnosticSwitches.ConvNoMatrix || _opt.SubgroupSize != 32
            || SmallFragments || n.Inputs[0].Rank != 4 || n.Inputs[1].Rank != 4)
        {
            return null;
        }

        DataType? type = _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) ? DataType.F16
            : _opt.Has(DeviceFeatures.MatrixBF16) && _opt.Has(DeviceFeatures.BFloat16) ? DataType.BF16 : null;
        var x = n.Inputs[0];
        var w = n.Inputs[1];
        if (type is null || w.Kind is not (ValueKind.Weight or ValueKind.Constant) || w.Type != DataType.F32 && w.Type != DataType.F16 && w.Type != DataType.BF16
            || x.Type != DataType.F32 && x.Type != DataType.F16 && x.Type != DataType.BF16 || x.Shape.Dims.Skip(1).Any(d => d.IsSymbolic)
            || w.Shape.Dims.Any(d => d.IsSymbolic))
        {
            return null;
        }

        long groups = n.Attributes.Get("groups", 1L);
        long channels = x.Shape[1].Fixed;
        long outputs = w.Shape[0].Fixed;
        long channelsPerGroup = w.Shape[1].Fixed;
        long outputsPerGroup = outputs / groups;
        bool depthwise = groups == channels && channelsPerGroup == 1;
        bool narrow = groups == 1 && outputs <= ConvNarrowMaxOutputs && !DiagnosticSwitches.NoNarrowConv;
        long columns = Bound(x.Shape[0]) * n.Output.Shape[2].Fixed * n.Output.Shape[3].Fixed;
        if (depthwise || narrow || channelsPerGroup % 8 != 0 || outputsPerGroup < ConvMatrixMinChannels || columns < ConvMatrixMinColumns)
        {
            return null;
        }

        int warpsM = outputsPerGroup >= 2 * ConvMatrixWarpRows ? 2 : 1;
        // 128-pixel tiles, or 64 when those would not fill the device (small maps at batch 1).
        long wideTiles = (columns + 127) / 128 * ((outputsPerGroup + warpsM * ConvMatrixWarpRows - 1) / (warpsM * ConvMatrixWarpRows)) * groups;
        var form = new ConvChannelsLastForm(type, warpsM, 2, wideTiles < TargetWorkgroups ? 2 : 4, 0);
        if (DiagnosticSwitches.ConvChannelsLastTile is [int tileWarpsM, int tileWarpsN, int tileFragN, ..] tile)
        {
            form = form with { WarpsM = tileWarpsM, WarpsN = tileWarpsN, FragN = tileFragN, Stages = tile.Length > 3 ? tile[3] : 0 };
        }
        else if (ConvAsyncCopies)
        {
            // The deepest ring of asynchronous copies that fits (four steps in flight on 99 KB devices with 128 x 128 tiles).
            form = form with { Stages = 4 };
            while (form.Stages > 2 && ConvChannelsLastSharedBytes(form) > _opt.MaxSharedMemory)
            {
                form = form with { Stages = form.Stages - 1 };
            }
        }

        return ConvChannelsLastSharedBytes(form) <= _opt.MaxSharedMemory && ConvChannelsLastEpilogueFits(form) ? form : null;
    }

    /// <summary>
    /// Marks the channel concatenations (rank 4, axis 1) whose only reader is a conv_cl convolution's input and whose parts are
    /// floats of whole 8-channel items: <see cref="LowerConcat"/> stages each part into its channel range of a channels-last
    /// 16-bit output instead of copying it, and the convolution reads that output directly. A part that is the only output
    /// read of a single-node unit converting or gathering other tensors (a Cast between float types, a Concat along the batch
    /// axis, e.g. a causal convolution's cached frames and the chunk's frames) is staged in place by that unit instead, from the
    /// unit's own inputs (<see cref="_channelsLastParts"/>), so neither the converted nor the gathered tensor is materialized.
    /// </summary>
    private void PlanChannelsLastConcats(List<LoweringUnit> units)
    {
        var readers = new Dictionary<Value, List<(Node Node, int Index)>>();
        var concats = new List<Node>();
        var singleNodeUnits = new Dictionary<Value, Node>();
        foreach (var unit in units)
        {
            if (unit.Node is { } single && unit.Fused is null && unit.Dual is null && unit.ConvHead is null)
            {
                singleNodeUnits[single.Output] = single;
            }
            else if (unit.Fused is [var only] && unit.FusedOutputs is [var output] && output == only.Output && unit.NormTail is null && unit.NormHead is null)
            {
                singleNodeUnits[only.Output] = only;
            }

            foreach (var node in unit.Nodes)
            {
                for (int i = 0; i < node.Inputs.Length; i++)
                {
                    if (!readers.TryGetValue(node.Inputs[i], out var list))
                    {
                        readers[node.Inputs[i]] = list = [];
                    }

                    list.Add((node, i));
                }

                if (node.Op.Name == "Concat")
                {
                    concats.Add(node);
                }
            }
        }

        foreach (var concat in concats)
        {
            var y = concat.Output;
            int axis = concat.Attributes.Get<int>("axis");
            if (y.Rank != 4 || (axis != 1 && axis != -3) || _segmentOutputs.Contains(y) || !readers.TryGetValue(y, out var uses) || uses.Count != 1
                || uses[0] is not ({ Op.Name: "Conv" }, 0) || ChooseConvChannelsLast(uses[0].Node) is not { } form
                || concat.Inputs.Any(i => i.Rank != 4 || i.Shape[1].IsSymbolic || i.Shape[1].Fixed % 8 != 0 || i.Type.IsQuantized))
            {
                continue;
            }

            _channelsLastConcats[y] = form.Type;
            long channel = 0;
            foreach (var part in concat.Inputs)
            {
                PlanChannelsLastPart(part, new ChannelsLastSlot(y, form.Type, channel, 0), readers, singleNodeUnits);
                channel += part.Shape[1].Fixed;
            }
        }
    }

    /// <summary>Registers <paramref name="part"/> as staged in place at <paramref name="slot"/> when its producer qualifies (see <see cref="PlanChannelsLastConcats"/>).</summary>
    private void PlanChannelsLastPart(Value part, ChannelsLastSlot slot, Dictionary<Value, List<(Node Node, int Index)>> readers, Dictionary<Value, Node> singleNodeUnits)
    {
        if (!singleNodeUnits.TryGetValue(part, out var producer) || _segmentOutputs.Contains(part) || readers[part].Count != 1 || _channelsLastParts.ContainsKey(part)
            || producer.Inputs.Any(i => i.Rank != 4 || i.Type != DataType.F32 && i.Type != DataType.F16 && i.Type != DataType.BF16))
        {
            return;
        }

        if (producer.Op.Name == "Cast" && part.Type == DataType.F32)
        {
            _channelsLastParts[part] = slot;
        }
        else if (producer.Op.Name == "Concat" && producer.Attributes.Get<int>("axis") is 0 or -4 && producer.Inputs.All(i => !i.Shape[0].IsSymbolic)
                 && producer.Inputs.Distinct().Count() == producer.Inputs.Length)
        {
            _channelsLastParts[part] = slot;
            long batch = slot.Batch;
            foreach (var input in producer.Inputs)
            {
                PlanChannelsLastPart(input, slot with { Batch = batch }, readers, singleNodeUnits);
                batch += input.Shape[0].Fixed;
            }
        }
    }

    /// <summary>
    /// Lowers the single node of a unit whose output is staged in place (<see cref="_channelsLastParts"/>): the inputs of a Cast or a
    /// batch Concat staged at the output's slot. False when <paramref name="node"/> has no such slot.
    /// </summary>
    private bool TryLowerChannelsLastPart(Node node)
    {
        if (!_channelsLastParts.TryGetValue(node.Output, out var slot))
        {
            return false;
        }

        long batch = slot.Batch;
        foreach (var input in node.Op.Name == "Cast" ? node.Inputs.Take(1) : node.Inputs)
        {
            if (!_channelsLastParts.ContainsKey(input))
            {
                StageChannelsLast(Ref(input), ChannelsLastTarget(slot with { Batch = batch }), slot.Channel, [node], $"{node.Op.Name.ToLowerInvariant()} {node.Output.Name} channels-last");
            }

            batch += input.Shape[0].Fixed;
        }

        return true;
    }

    /// <summary>The channels-last concatenation output of <paramref name="slot"/> (allocated on first use), from its batch offset on.</summary>
    private TensorRef ChannelsLastTarget(ChannelsLastSlot slot)
    {
        var y = _refs.TryGetValue(slot.Concat, out var existing) ? existing : ProduceChannelsLast(slot.Concat, slot.Type);
        return y with { ByteOffset = y.ByteOffset + slot.Batch * y.Strides[0] * y.Type.ElementSize };
    }

    /// <summary>The concatenation planned by <see cref="PlanChannelsLastConcats"/>: each part not staged in place is staged into its channel range.</summary>
    private void LowerConcatChannelsLast(Node n, DataType type)
    {
        long offset = 0;
        foreach (var input in n.Inputs)
        {
            if (!_channelsLastParts.ContainsKey(input))
            {
                StageChannelsLast(Ref(input), ChannelsLastTarget(new ChannelsLastSlot(n.Output, type, offset, 0)), offset, [n], $"concat {n.Output.Name} channels-last");
            }

            offset += input.Shape[1].Fixed;
        }

        _ = ChannelsLastTarget(new ChannelsLastSlot(n.Output, type, 0, 0));
    }

    /// <summary>A channels-last 16-bit (<paramref name="type"/>) tensor for <paramref name="v"/>, allocated in the arena under its value.</summary>
    private TensorRef ProduceChannelsLast(Value v, DataType type)
    {
        var shape = BoundShape(v.Shape);
        long bytes = type.ByteSize(shape.Aggregate(1L, (a, b) => a * b));
        long offset = _alloc.Allocate(v, Math.Max(bytes, 16));
        var r = new TensorRef(_arena, offset, type, shape, ChannelsLastStrides(shape), v.Shape.Symbol);
        _refs[v] = r;
        return r;
    }

    /// <summary>Element strides of the channels-last layout of [N, C, H, W]: channel stride 1.</summary>
    private static long[] ChannelsLastStrides(long[] shape) => [shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1]];

    /// <summary>True when <paramref name="x"/> is channels-last <paramref name="type"/> with 16-byte aligned 8-channel items.</summary>
    private static bool IsChannelsLast(TensorRef x, DataType type)
        => x.Type == type && x.Rank == 4 && x.Strides[1] == 1 && x.Strides[0] % 8 == 0 && x.Strides[2] % 8 == 0 && x.Strides[3] % 8 == 0
           && x.ByteOffset % 16 == 0;

    /// <summary>Emits conv_cl_stage: <paramref name="x"/> [N, C, H, W] (any strides, f32 / f16 / bf16) into channels [offset, offset + C) of <paramref name="y"/>.</summary>
    private void StageChannelsLast(TensorRef x, TensorRef y, long channelOffset, Node[] nodes, string name)
    {
        long c = x.Shape[1];
        long pixels = x.Shape[2] * x.Shape[3];
        long perThread = c % 16 == 0 ? 16 : 8;
        int format = WeightFormat(x.Type) ?? throw new NotSupportedException($"channels-last staging of {x.Type}");
        var kernel = Lib("conv_cl.slang", "conv_cl_stage", Macros(("C", c), ("H", x.Shape[2]), ("W", x.Shape[3]),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("XFMT", format),
            ("OS_N", y.Strides[0]), ("OS_H", y.Strides[2]), ("OS_W", y.Strides[3]), ("C_OFF", channelOffset), ("V", perThread),
            ("XBF16", y.Type == DataType.BF16)));
        long items = pixels * (c / perThread);
        var source = x;
        y = y with { Shape = [x.Shape[0], y.Shape[1], y.Shape[2], y.Shape[3]], Symbol = x.Symbol };
        Emit(name, nodes, [Always(kernel, b => new Grid(Grid.Div(Rows(b, source) * items, 256)))],
            [Pointer(x), Pointer(y), CountArg(x, pixels), CountArg(x, items)], [x], [y]);
    }

    /// <summary>
    /// The channels-last weights of constant conv weight <paramref name="weight"/> ([O, CG, KH, KW], f32 / f16 / bf16): [O, KH·KW·CG]
    /// of <paramref name="type"/>, each row reordered tap-major (k = tap · CG + channel), converted on the host when uploaded.
    /// </summary>
    private TensorRef ChannelsLastWeights(Value weight, DataType type)
    {
        if (_channelsLastWeights.TryGetValue((weight, type), out var cached))
        {
            return cached;
        }

        long rows = weight.Shape[0].Fixed;
        long channels = weight.Shape[1].Fixed;
        long taps = weight.Shape[2].Fixed * weight.Shape[3].Fixed;
        var sourceType = weight.Type;
        byte[] Original()
        {
            if (weight.Kind == ValueKind.Constant)
            {
                return weight.ConstantData!;
            }

            var bytes = new byte[sourceType.ByteSize(rows * channels * taps)];
            weight.Weight!.Source.Read(0, bytes);
            return bytes;
        }

        string name = $"{weight.Name}:{type.Name}:tap-major";
        WeightInfo Create() => new()
        {
            Name = name,
            Type = type,
            Shape = new Shape(rows, channels * taps),
            Source = new TransientWeightSource(type.ByteSize(rows * channels * taps), () => HalfTapMajor.Convert(Original(), sourceType, type, rows, channels, taps)),
        };

        var info = weight.Kind == ValueKind.Weight ? DerivedWeightCache.Get(weight.Weight!, (type.Name, rows, channels, taps), Create) : Create();
        var r = new TensorRef(NewBuffer(BufferKind.Weight, info.Name, info.ByteSize, w: info), 0, type, [rows, channels * taps], [channels * taps, 1], null);
        _channelsLastWeights[(weight, type)] = r;
        return r;
    }

    /// <summary>Lowers convolution <paramref name="n"/> (with <paramref name="epilogue"/>'s group in its store) through conv_cl.</summary>
    private void LowerConvChannelsLast(Node n, Node[] nodes, ConvEpilogue? fused, ConvEpilogueCode? epilogue, ConvChannelsLastForm form, string fusedLabel)
    {
        var input = Ref(n.Inputs[0]);
        var x = input;
        if (!IsChannelsLast(input, form.Type))
        {
            x = Temp(form.Type, input.Shape, input.Symbol) with { Strides = ChannelsLastStrides(input.Shape) };
            StageChannelsLast(input, x, 0, [n], $"conv {n.Output.Name} stage channels-last");
        }

        var w = ChannelsLastWeights(n.Inputs[1], form.Type);
        TensorRef? bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "conv-bias") : null;
        var a = n.Attributes;
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        long G = a.Get("groups", 1L);
        long O = n.Inputs[1].Shape[0].Fixed;
        long CG = n.Inputs[1].Shape[1].Fixed;
        long KH = n.Inputs[1].Shape[2].Fixed;
        long KW = n.Inputs[1].Shape[3].Fixed;
        long OG = O / G;
        var y = Produce(fused?.Output ?? n.Output);
        long HW = y.Shape[2] * y.Shape[3];
        long K = CG * KH * KW;
        long bound = x.Symbol is null ? x.Shape[0] : Bound(Dim.Sym(x.Symbol));
        var macros = new List<(string, object)>
        {
            ("C", x.Shape[1]), ("H", x.Shape[2]), ("W", x.Shape[3]), ("O", O), ("HO", y.Shape[2]), ("WO", y.Shape[3]), ("CG", CG), ("OG", OG), ("G", G),
            ("KH", KH), ("KW", KW), ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("DH", dl[0]), ("DW", dl[1]),
            ("XS_N", x.Strides[0]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("NOPAD", pd.All(p => p == 0)), ("XBF16", form.Type == DataType.BF16),
            ("WARPS_M", form.WarpsM), ("WARPS_N", form.WarpsN), ("FRAG_M", 4), ("FRAG_N", form.FragN), ("CL_ASYNC", form.Stages > 0),
            ("CL_STAGES", form.Buffers),
        };
        string name = $"conv {n.Output.Name} [{x.Shape[1]}→{O} {KH}x{KW} s{st[0]} g{G}]{fusedLabel} {form.Type.Name} channels-last";
        EmitConvMatrix(name, nodes, "conv_cl.slang", "conv_cl", macros, epilogue, x, w, [], [], bias, y, G, OG, HW, K, bound, form.TileRows, form.TileColumns);
    }
}
