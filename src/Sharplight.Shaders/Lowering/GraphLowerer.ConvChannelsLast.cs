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
    }

    /// <summary>Most parts of a concatenated input conv_cl reads in place (<c>X_PARTS</c>).</summary>
    private const int ConvChannelsLastMaxParts = 4;

    /// <summary>
    /// Concatenations lowered as channels-last 16-bit stagings of their parts, with their type and whether the convolution reads the
    /// parts in place (see <see cref="PlanChannelsLastConcats"/>).
    /// </summary>
    private readonly Dictionary<Value, (DataType Type, bool InPlace)> _channelsLastConcats = [];

    /// <summary>Parts of those concatenations (or of their parts) that the unit producing them stages, by value.</summary>
    private readonly Dictionary<Value, ChannelsLastSlot> _channelsLastParts = [];

    /// <summary>
    /// Where a part is staged: the concatenation's output, the value whose channels-last tensor holds it (the output, or the part for
    /// concatenations read in place), its type, and the part's channel and batch offsets there.
    /// </summary>
    private readonly record struct ChannelsLastSlot(Value Concat, Value Owner, DataType Type, long Channel, long Batch);

    /// <summary>The own channels-last tensors of parts read in place, by part.</summary>
    private readonly Dictionary<Value, TensorRef> _channelsLastOwned = [];

    /// <summary>Parts of concatenations read in place that are slices of another value (through Slice and After), with that value.</summary>
    private readonly Dictionary<Value, Value> _channelsLastViews = [];

    /// <summary>The (source, concatenation output) pairs of <see cref="_channelsLastViews"/>: the copy of the source must outlive the concatenation's reader.</summary>
    private readonly List<(Value Source, Value Concat)> _channelsLastViewReads = [];

    /// <summary>Values whose parts are read from one channels-last copy (they cover the value at least once).</summary>
    private readonly HashSet<Value> _channelsLastSources = [];

    /// <summary>The channels-last copies of <see cref="_channelsLastSources"/>, while their values live.</summary>
    private readonly Dictionary<Value, TensorRef> _channelsLastCopies = [];

    /// <summary>The parts a convolution reads in place for a concatenated input, by the concatenation's output.</summary>
    private readonly Dictionary<Value, List<TensorRef>> _channelsLastInputs = [];

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
            form = form with { WarpsM = tileWarpsM, WarpsN = tileWarpsN, FragN = tileFragN, Stages = tile.Length > 3 && ConvAsyncCopies ? tile[3] : 0 };
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
    /// floats of whole 8-channel items, so they are not copied as such:
    /// <list type="bullet">
    /// <item>Two to four parts of one shape into a one-group convolution (the stacked frame windows of a causal 3-D convolution):
    /// the convolution reads the parts in place (<c>X_PARTS</c>), each a channels-last 16-bit tensor. A part that is a slice of a
    /// value the parts of these concatenations cover at least once in total (the frames of a chunk, which the windows of a causal
    /// convolution's pieces take one tap after the other) is a view of that value's channels-last copy, staged once
    /// (<see cref="_channelsLastSources"/>); any other part is staged into a tensor of its own.</item>
    /// <item>Otherwise the output is one channels-last 16-bit tensor and each part is staged into its channel range.</item>
    /// </list>
    /// A part that is the only output read of a single-node unit converting or gathering other tensors (a Cast between float types,
    /// a Concat along the batch axis, e.g. a causal convolution's cached frames and the chunk's frames) is staged by that unit from
    /// its own inputs (<see cref="_channelsLastParts"/>), so neither the converted nor the gathered tensor is materialized.
    /// </summary>
    private void PlanChannelsLastConcats(List<LoweringUnit> units)
    {
        var readers = new Dictionary<Value, List<(Node Node, int Index)>>();
        var concats = new List<Node>();
        var singleNodeUnits = new Dictionary<Value, Node>();
        var producers = new Dictionary<Value, Node>();
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
                foreach (var output in node.Outputs)
                {
                    producers[output] = node;
                }

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

        var coverage = new Dictionary<Value, long>();
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

            var first = concat.Inputs[0].Shape;
            bool inPlace = concat.Inputs.Length is >= 2 and <= ConvChannelsLastMaxParts && uses[0].Node.Attributes.Get("groups", 1L) == 1
                           && concat.Inputs.All(i => i.Shape.Dims.SequenceEqual(first.Dims));
            _channelsLastConcats[y] = (form.Type, inPlace);
            long channel = 0;
            foreach (var part in concat.Inputs)
            {
                var slot = inPlace ? new ChannelsLastSlot(y, part, form.Type, 0, 0) : new ChannelsLastSlot(y, y, form.Type, channel, 0);
                if (!PlanChannelsLastPart(part, slot, readers, singleNodeUnits) && inPlace && ViewSource(part, producers) is { } source && source != part)
                {
                    _channelsLastViews[part] = source;
                    _channelsLastViewReads.Add((source, y));
                    coverage[source] = coverage.GetValueOrDefault(source) + part.Shape.Dims.Aggregate(1L, (a, d) => a * Bound(d));
                }

                channel += part.Shape[1].Fixed;
            }
        }

        foreach (var (source, elements) in coverage)
        {
            if (elements >= source.Shape.Dims.Aggregate(1L, (a, d) => a * Bound(d)) && source.Rank == 4
                && (source.Type == DataType.F32 || source.Type == DataType.F16 || source.Type == DataType.BF16))
            {
                _channelsLastSources.Add(source);
            }
        }
    }

    /// <summary>The value <paramref name="part"/> is a slice of (through Slice and After), or <paramref name="part"/> itself.</summary>
    private static Value ViewSource(Value part, Dictionary<Value, Node> producers)
    {
        while (producers.TryGetValue(part, out var producer) && producer.Op.Name is "Slice" or "After")
        {
            part = producer.Inputs[0];
        }

        return part;
    }

    /// <summary>
    /// Keeps the values whose channels-last copies the planned concatenations read (<see cref="_channelsLastSources"/>) alive until
    /// the last convolution reading them (the copy is released with its value, see <see cref="ReleaseRootScratch"/>).
    /// </summary>
    private void ExtendChannelsLastSources()
    {
        foreach (var (source, concat) in _channelsLastViewReads)
        {
            var root = RootOf(source);
            if (_channelsLastSources.Contains(source) && _lastUse.TryGetValue(RootOf(concat), out int last) && _lastUse.TryGetValue(root, out int current)
                && last > current)
            {
                _lastUse[root] = last;
            }
        }
    }

    /// <summary>
    /// Registers <paramref name="part"/> as staged by its producer at <paramref name="slot"/> when the producer qualifies (see
    /// <see cref="PlanChannelsLastConcats"/>); false otherwise.
    /// </summary>
    private bool PlanChannelsLastPart(Value part, ChannelsLastSlot slot, Dictionary<Value, List<(Node Node, int Index)>> readers, Dictionary<Value, Node> singleNodeUnits)
    {
        if (!singleNodeUnits.TryGetValue(part, out var producer) || _segmentOutputs.Contains(part) || readers[part].Count != 1 || _channelsLastParts.ContainsKey(part)
            || producer.Inputs.Any(i => i.Rank != 4 || i.Type != DataType.F32 && i.Type != DataType.F16 && i.Type != DataType.BF16))
        {
            return false;
        }

        if (producer.Op.Name == "Cast" && part.Type == DataType.F32)
        {
            _channelsLastParts[part] = slot;
            return true;
        }

        if (producer.Op.Name == "Concat" && producer.Attributes.Get<int>("axis") is 0 or -4 && producer.Inputs.All(i => !i.Shape[0].IsSymbolic)
            && producer.Inputs.Distinct().Count() == producer.Inputs.Length)
        {
            _channelsLastParts[part] = slot;
            long batch = slot.Batch;
            foreach (var input in producer.Inputs)
            {
                PlanChannelsLastPart(input, slot with { Batch = batch }, readers, singleNodeUnits);
                batch += input.Shape[0].Fixed;
            }

            return true;
        }

        return false;
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

    /// <summary>
    /// The channels-last tensor of <paramref name="slot"/> (allocated on first use) from its batch offset on: the concatenation's
    /// output, or the part's own tensor (released with the concatenation's output).
    /// </summary>
    private TensorRef ChannelsLastTarget(ChannelsLastSlot slot)
    {
        TensorRef target;
        if (slot.Owner == slot.Concat)
        {
            target = _refs.TryGetValue(slot.Concat, out var existing) ? existing : ProduceChannelsLast(slot.Concat, slot.Type);
        }
        else if (!_channelsLastOwned.TryGetValue(slot.Owner, out target!))
        {
            target = ChannelsLastScratch(slot.Owner.Shape, slot.Type, slot.Concat);
            _channelsLastOwned[slot.Owner] = target;
        }

        return target with { ByteOffset = target.ByteOffset + slot.Batch * target.Strides[0] * target.Type.ElementSize };
    }

    /// <summary>A channels-last tensor of <paramref name="shape"/> in the arena, released with <paramref name="owner"/>'s root.</summary>
    private TensorRef ChannelsLastScratch(Shape shape, DataType type, Value owner)
    {
        var bounds = BoundShape(shape);
        var allocation = new object();
        long offset = _alloc.Allocate(allocation, Math.Max(16, type.ByteSize(bounds.Aggregate(1L, (a, b) => a * b))));
        var root = RootOf(owner);
        if (!_rootScratch.TryGetValue(root, out var list))
        {
            _rootScratch[root] = list = [];
        }

        list.Add(allocation);
        return new TensorRef(_arena, offset, type, bounds, ChannelsLastStrides(bounds), shape.Symbol);
    }

    /// <summary>
    /// The concatenation planned by <see cref="PlanChannelsLastConcats"/>. Read in place: the parts' channels-last tensors (the
    /// copies of their sources, their own staged tensors), or, when their strides differ, one tensor they are staged into. Else
    /// each part not staged by its producer is staged into its channel range.
    /// </summary>
    private void LowerConcatChannelsLast(Node n, DataType type, bool inPlace)
    {
        if (!inPlace)
        {
            long offset = 0;
            foreach (var input in n.Inputs)
            {
                if (!_channelsLastParts.ContainsKey(input))
                {
                    StageChannelsLast(Ref(input), ChannelsLastTarget(new ChannelsLastSlot(n.Output, n.Output, type, offset, 0)), offset, [n], $"concat {n.Output.Name} channels-last");
                }

                offset += input.Shape[1].Fixed;
            }

            _ = ChannelsLastTarget(new ChannelsLastSlot(n.Output, n.Output, type, 0, 0));
            return;
        }

        var parts = new List<TensorRef>();
        foreach (var input in n.Inputs)
        {
            var slot = new ChannelsLastSlot(n.Output, input, type, 0, 0);
            if (_channelsLastParts.ContainsKey(input))
            {
                parts.Add(ChannelsLastTarget(slot));
            }
            else if (_channelsLastViews.TryGetValue(input, out var source) && _channelsLastSources.Contains(source)
                     && ChannelsLastView(Ref(input), source, type, n) is { } view)
            {
                parts.Add(view);
            }
            else
            {
                var own = ChannelsLastTarget(slot);
                StageChannelsLast(Ref(input), own, 0, [n], $"concat {n.Output.Name} part channels-last");
                parts.Add(own);
            }
        }

        if (parts.All(p => IsChannelsLast(p, type) && p.Strides.SequenceEqual(parts[0].Strides)))
        {
            _channelsLastInputs[n.Output] = parts;
            return;
        }

        var y = ProduceChannelsLast(n.Output, type);
        long channel = 0;
        foreach (var part in parts)
        {
            StageChannelsLast(part, y, channel, [n], $"concat {n.Output.Name} channels-last");
            channel += part.Shape[1];
        }
    }

    /// <summary>
    /// <paramref name="part"/> (a slice of <paramref name="source"/>'s dense tensor: same strides) as a view of the source's
    /// channels-last copy, staged on first use and released with the source; null when it is no such slice.
    /// </summary>
    private TensorRef? ChannelsLastView(TensorRef part, Value source, DataType type, Node concat)
    {
        var whole = Ref(source);
        if (whole.Rank != 4 || !whole.IsDense || part.Buffer != whole.Buffer || part.Type != whole.Type || !part.Strides.SequenceEqual(whole.Strides)
            || part.ByteOffset < whole.ByteOffset)
        {
            return null;
        }

        long element = (part.ByteOffset - whole.ByteOffset) / whole.Type.ElementSize;
        long n = element / whole.Strides[0];
        long c = element % whole.Strides[0] / whole.Strides[1];
        long h = element % whole.Strides[1] / whole.Strides[2];
        long w = element % whole.Strides[2];
        if (c % 8 != 0)
        {
            return null;
        }

        if (!_channelsLastCopies.TryGetValue(source, out var copy))
        {
            copy = ChannelsLastScratch(source.Shape, type, source);
            StageChannelsLast(whole, copy, 0, [concat], $"{source.Name} channels-last copy");
            _channelsLastCopies[source] = copy;
        }

        long offset = n * copy.Strides[0] + c + h * copy.Strides[2] + w * copy.Strides[3];
        return copy with { ByteOffset = copy.ByteOffset + offset * type.ElementSize, Shape = part.Shape, Symbol = part.Symbol };
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
        TensorRef x;
        var parts = _channelsLastInputs.GetValueOrDefault(n.Inputs[0]) ?? [];
        if (parts.Count > 0)
        {
            x = parts[0];
        }
        else
        {
            x = Ref(n.Inputs[0]);
            if (!IsChannelsLast(x, form.Type))
            {
                var input = x;
                x = Temp(form.Type, input.Shape, input.Symbol) with { Strides = ChannelsLastStrides(input.Shape) };
                StageChannelsLast(input, x, 0, [n], $"conv {n.Output.Name} stage channels-last");
            }
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
        long channelsTotal = parts.Count > 0 ? parts.Count * x.Shape[1] : x.Shape[1];
        var macros = new List<(string, object)>
        {
            ("C", channelsTotal), ("H", x.Shape[2]), ("W", x.Shape[3]), ("O", O), ("HO", y.Shape[2]), ("WO", y.Shape[3]), ("CG", CG), ("OG", OG), ("G", G),
            ("KH", KH), ("KW", KW), ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("DH", dl[0]), ("DW", dl[1]),
            ("XS_N", x.Strides[0]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("NOPAD", pd.All(p => p == 0)), ("XBF16", form.Type == DataType.BF16),
            ("WARPS_M", form.WarpsM), ("WARPS_N", form.WarpsN), ("FRAG_M", 4), ("FRAG_N", form.FragN), ("CL_ASYNC", form.Stages > 0),
            ("CL_STAGES", form.Buffers), ("X_PARTS", Math.Max(1, parts.Count)), ("PART_C", x.Shape[1]),
        };
        string name = $"conv {n.Output.Name} [{channelsTotal}→{O} {KH}x{KW} s{st[0]} g{G}]{fusedLabel} {form.Type.Name} channels-last{(parts.Count > 0 ? $" {parts.Count} parts" : "")}";
        var extra = parts.Skip(1).ToArray();
        EmitConvMatrix(name, nodes, "conv_cl.slang", "conv_cl", macros, epilogue, x, w, [.. extra.Select(p => (KernelArg)Pointer(p))], extra, bias, y, G, OG, HW, K, bound,
            form.TileRows, form.TileColumns);
    }
}
