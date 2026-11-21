using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>
/// A graph compiled onto one or more devices. Executes its segments in order and moves boundary
/// activations between devices through host memory (activations are small compared to weights).
/// </summary>
public sealed class CompiledModel : IDisposable
{
    private readonly ICompiledSegment[] _segments;
    /// <summary>Per segment: the GPU compilation that streams its weights and runs it for large batches (<see cref="PlacementPlan.StreamedSegments"/>), or null.</summary>
    private readonly ICompiledSegment?[] _streamed;
    private readonly int _streamMinRows;
    /// <summary>Streamed-weight memories (<see cref="PlacementOptions.StreamBuffers"/>): streamed segment k uses memory k mod this.</summary>
    private readonly int _streamBuffers;
    /// <summary>Per segment with a streamed compilation: its streamed-weight memory (the compilations alternate between them in segment order).</summary>
    private readonly int[] _streamBufferOf;
    /// <summary>
    /// Per streamed-weight memory: the segment whose weights it holds (its last copy), or -1. A segment whose weights are still there runs on
    /// the GPU for small batches too (decode), without a copy: the last streamed segment of each memory is copied in at load and is the last
    /// one every streaming pass copies, so the memory the GPU keeps for streaming holds one layer's experts between prefills.
    /// </summary>
    private readonly int[] _residentIn;
    private readonly bool _keepsResident = Environment.GetEnvironmentVariable("SHARPLIGHT_STREAM_RESIDENT") is not ("0" or "false");
    /// <summary>
    /// Per CPU segment of routed experts: its GPU compilation that copies in only the experts a pass selects
    /// (<see cref="CompileOptions.SparseExpertSlots"/>), or null. Small passes (decode) run the slots of every row partly there and partly on the
    /// CPU at the same time (<see cref="RunSplit"/>).
    /// </summary>
    private readonly ICompiledSegment?[] _sparse;
    /// <summary>Per segment with a sparse compilation: the expert ids input of its expert products.</summary>
    private readonly Value?[] _sparseIds;
    /// <summary>
    /// Per segment with a sparse compilation whose slots without an expert still produce values (it gathers per-expert table rows by the ids,
    /// e.g. biases): the input of the slot weights that scale every slot before the sum, zeroed for the other side's slots in a split pass.
    /// </summary>
    private readonly Value?[] _slotWeights;
    /// <summary>Per segment with a sparse compilation and rows per pass: how many of a row's slots (expert choices) the GPU takes.</summary>
    private readonly Dictionary<(int Segment, long Rows), SplitBalance> _balances = [];
    /// <summary>Per segment: reused id buffers of the GPU's and the CPU's share of a split pass, and the GPU's outputs.</summary>
    private readonly Dictionary<int, SplitBuffers> _splitBuffers = [];

    /// <summary>
    /// Rows per pass up to which the routed experts of a CPU layer run split between the CPU and the GPU (<c>SHARPLIGHT_SPARSE_EXPERT_ROWS</c>,
    /// default 4; 0 turns the split off): the GPU copies the few experts its share of the slots selects (a decode step of Qwen3.5-122B: 8 of
    /// 256, 21 MB) while the CPU computes the others.
    /// </summary>
    internal static int SparseExpertRows { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_SPARSE_EXPERT_ROWS"), out int rows) && rows >= 0 ? rows : 4;

    /// <summary>
    /// Experts per slot of a sparse compilation at least (a pass of <see cref="SparseExpertRows"/> rows selects at most a quarter of the bank):
    /// with fewer the copies of the selected experts approach a copy of the whole bank, which the streamed compilation already makes.
    /// </summary>
    internal const int SparseMinExpertsPerSlot = 4;

    /// <summary>A fixed number of slots per row for the GPU in split passes (<c>SHARPLIGHT_SPARSE_GPU_SLOTS</c>), or null to adapt it.</summary>
    private static int? DiagnosticSparseGpuSlots { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_SPARSE_GPU_SLOTS"), out int slots) && slots >= 0 ? slots : null;

    /// <summary>With <c>SHARPLIGHT_SEGMENT_TIMING=1</c>: per segment, the host time of its runs (inputs, execution, outputs), the run count and the device time its provider measured.</summary>
    private readonly (long Ticks, int Runs, bool Streamed, double GpuMs)[]? _segmentTimes;
    /// <summary>With <c>SHARPLIGHT_SEGMENT_TIMING=1</c>: per segment, the split passes' GPU slots, CPU time and wait for the GPU, summed.</summary>
    private readonly (long GpuSlots, double CpuMs, double WaitMs, int Runs)[]? _splitTimes;
    private static readonly HashSet<Value> s_noSkips = [];
    private readonly Dictionary<Value, byte[]> _hostValues = [];
    /// <summary>Per segment: the inputs it reads from an earlier segment's device memory (<see cref="IDeviceForwardingSegment"/>).</summary>
    private readonly HashSet<Value>[] _forwardedInputs;
    /// <summary>Per segment: the outputs that only reach device-forwarding consumers, so they are never downloaded.</summary>
    private readonly HashSet<Value>[] _deviceOnlyOutputs;
    private readonly Dictionary<Value, string[]> _stringInputs = [];
    private RunParameters? _lastRun;

    /// <summary>The graph that was compiled.</summary>
    public ComputeGraph Graph { get; }
    /// <summary>How the graph was distributed over devices.</summary>
    public PlacementPlan Plan { get; }
    /// <summary>The options the graph was compiled with.</summary>
    public CompileOptions Options { get; }
    /// <summary>The compiled segments in execution order.</summary>
    public IReadOnlyList<ICompiledSegment> Segments => _segments;

    /// <summary>
    /// Number of CPU segments of routed experts that also have a GPU compilation streaming their weights, which runs them for large batches
    /// (<see cref="PlacementPlan.StreamedSegments"/>, <see cref="PlacementOptions.StreamExpertsMinRows"/>).
    /// </summary>
    public int StreamedSegmentCount => _streamed.Count(s => s is not null);

    /// <summary>
    /// Number of those CPU segments that also have a GPU compilation copying in only the selected experts (<see cref="CompileOptions.SparseExpertSlots"/>):
    /// passes of up to <c>SHARPLIGHT_SPARSE_EXPERT_ROWS</c> rows (default 4) run their slots split between it and the CPU.
    /// </summary>
    public int SparseSegmentCount => _sparse.Count(s => s is not null);

    private CompiledModel(ComputeGraph graph, PlacementPlan plan, CompileOptions options, ICompiledSegment[] segments, ICompiledSegment?[] streamed, int streamMinRows, int streamBuffers,
        ICompiledSegment?[] sparse)
    {
        _sparse = sparse;
        _sparseIds = new Value?[segments.Length];
        _slotWeights = new Value?[segments.Length];
        for (int i = 0; i < sparse.Length; i++)
        {
            if (sparse[i] is not null && SparseExpertIds(segments[i].Segment) is { } ids)
            {
                _sparseIds[i] = ids;
                _slotWeights[i] = SlotWeights(segments[i].Segment, ids).Weights;
            }
        }

        _streamBuffers = streamBuffers;
        _streamBufferOf = new int[streamed.Length];
        int ordinal = 0;
        for (int i = 0; i < streamed.Length; i++)
        {
            if (streamed[i] is not null)
            {
                _streamBufferOf[i] = ordinal++ % streamBuffers;
            }
        }

        Graph = graph;
        Plan = plan;
        Options = options;
        _segments = segments;
        _streamed = streamed;
        _streamMinRows = streamMinRows;
        (_forwardedInputs, _deviceOnlyOutputs) = ForwardOnDevice(graph, segments);
        _segmentTimes = Environment.GetEnvironmentVariable("SHARPLIGHT_SEGMENT_TIMING") is "1" or "true" ? new (long, int, bool, double)[segments.Length] : null;
        _splitTimes = _segmentTimes is null ? null : new (long, double, double, int)[segments.Length];
        _residentIn = new int[streamBuffers];
        _residentIn.AsSpan().Fill(-1);
        if (_keepsResident && streamMinRows > 0)
        {
            // The last streamed segment of each memory: its weights stay there between streaming passes.
            for (int i = streamed.Length - 1; i >= 0; i--)
            {
                if (streamed[i] is IStreamedSegment last && _residentIn[_streamBufferOf[i]] < 0)
                {
                    Stream(i, last);
                }
            }
        }
    }

    /// <summary>Starts the copy of streamed segment <paramref name="i"/>'s weights into its memory, unless the memory still holds them.</summary>
    private void Stream(int i, IStreamedSegment segment)
    {
        int buffer = _streamBufferOf[i];
        segment.WeightsResident = _residentIn[buffer] == i;
        segment.BeginStreaming();
        _residentIn[buffer] = i;
    }

    /// <summary>
    /// The GPU compilations of the CPU segments that stream their weights (<see cref="StreamedSegmentCount"/>), in segment order: they run the
    /// segment for large batches, and for small batches while their weights are still in the streamed-weight memory.
    /// </summary>
    public IEnumerable<ICompiledSegment> StreamedSegments => _streamed.OfType<ICompiledSegment>();

    /// <summary>
    /// Connects boundary values whose consumer segment can read them from the producer segment's device memory, and finds the outputs
    /// that then need no download. <c>SHARPLIGHT_DEVICE_FORWARDING=0</c> passes every boundary value through host memory.
    /// </summary>
    private static (HashSet<Value>[] Forwarded, HashSet<Value>[] DeviceOnly) ForwardOnDevice(ComputeGraph graph, ICompiledSegment[] segments)
    {
        var forwarded = segments.Select(_ => new HashSet<Value>()).ToArray();
        var deviceOnly = segments.Select(_ => new HashSet<Value>()).ToArray();
        if (segments.Length < 2 || Environment.GetEnvironmentVariable("SHARPLIGHT_DEVICE_FORWARDING") is "0" or "false")
        {
            return (forwarded, deviceOnly);
        }

        var producer = new Dictionary<Value, int>();
        var hostRead = new HashSet<Value>(graph.Outputs.Values);
        for (int i = 0; i < segments.Length; i++)
        {
            foreach (var v in segments[i].Segment.Inputs)
            {
                if (v.Kind != ValueKind.Intermediate || !producer.TryGetValue(v, out int p))
                {
                    continue;
                }

                if (segments[i] is IDeviceForwardingSegment consumer && consumer.TryForwardFrom(v, segments[p]))
                {
                    forwarded[i].Add(v);
                }
                else
                {
                    hostRead.Add(v);
                }
            }

            foreach (var v in segments[i].Segment.Outputs)
            {
                producer[v] = i;
            }
        }

        foreach (var (v, p) in producer)
        {
            if (!hostRead.Contains(v) && forwarded.Any(f => f.Contains(v)))
            {
                deviceOnly[p].Add(v);
            }
        }

        return (forwarded, deviceOnly);
    }

    /// <summary>
    /// Places the graph and compiles every segment on its device.
    /// </summary>
    /// <param name="graph">The graph to compile.</param>
    /// <param name="placement">Devices, fallback, memory reserve and compile options.</param>
    /// <returns>The compiled model; the caller disposes it.</returns>
    /// <exception cref="ArgumentException">No device was given.</exception>
    /// <exception cref="InsufficientMemoryException">A placement group fits on no device and no fallback device is configured.</exception>
    /// <exception cref="NotSupportedException">A node is supported neither by its device nor by the fallback device.</exception>
    public static CompiledModel Compile(ComputeGraph graph, PlacementOptions placement)
    {
        placement = WithDynamicBounds(graph, placement);
        for (int attempt = 0; ; attempt++)
        {
            var shortfall = new StrongBox<long>();
            try
            {
                return CompilePlaced(graph, placement, retryOnStreamingOom: attempt < MaxMemoryRetries, shortfall);
            }
            catch (DeviceOutOfMemoryException e) when (placement.Fallback is not null && attempt < MaxMemoryRetries)
            {
                // The plan's estimate of what a device holds besides weights and state (activations, library workspaces, driver
                // overhead, other processes' allocations since the plan) fell short: plan again with what the rest of the plan
                // still needed beyond the free memory kept free in addition (at least MinRetryReserve when that is unknown), so
                // more of the model goes to the fallback.
                long reserve = placement.ReservePerDevice + (shortfall.Value > 0 ? shortfall.Value + RetryMargin : Math.Max(MinRetryReserve, e.RequestedBytes));
                if (Environment.GetEnvironmentVariable("SHARPLIGHT_LOAD_TIMING") is "1" or "true")
                {
                    Console.Error.WriteLine($"[load] {e.Device}: out of memory for {e.What} ({e.RequestedBytes >> 20} MiB); placing '{graph.Name}' again with {reserve >> 20} MiB reserved");
                }

                placement = placement.WithReserve(reserve);
            }
        }
    }

    /// <summary>How often <see cref="Compile"/> places a graph again with a larger reserve after a device ran out of memory.</summary>
    private const int MaxMemoryRetries = 3;

    /// <summary>The extra memory a placement after an out-of-memory failure reserves on every device when the shortfall is unknown.</summary>
    private const long MinRetryReserve = 256L << 20;

    /// <summary>Added to a measured shortfall (<see cref="MemoryShortfall"/>) when a graph is placed again after an out-of-memory failure.</summary>
    private const long RetryMargin = 64L << 20;

    /// <summary>Per segment still to compile in <see cref="MemoryShortfall"/>: its input and output buffers, stream and launch recordings.</summary>
    private const long SegmentOverheadEstimate = 32L << 20;

    /// <summary>
    /// After an out-of-memory failure during <see cref="CompilePlaced"/>: how many bytes the accelerators lacked for the rest of the plan, the
    /// weights and state of the segments not compiled yet (and the streamed and sparse expert memories when they were not reached) less the
    /// memory still free, the largest over the devices (0 when it does not show a lack).
    /// </summary>
    private static long MemoryShortfall(PlacementPlan plan, CompileOptions options, ICompiledSegment?[] compiled, bool twinsCompiled)
    {
        long worst = 0;
        foreach (var device in plan.Slots.Where(s => s.Device.Info.Kind != DeviceKind.Cpu).Select(s => s.Device).Distinct())
        {
            var weights = new HashSet<Models.WeightInfo>();
            var states = new HashSet<Models.StateDefinition>();
            long remaining = 0;
            for (int i = 0; i < plan.Segments.Count; i++)
            {
                var segment = plan.Segments[i];
                if (compiled[i] is not null || plan.Slots[segment.DeviceSlot].Device != device)
                {
                    continue;
                }

                remaining += SegmentOverheadEstimate;
                foreach (var v in segment.Nodes.SelectMany(n => n.Inputs))
                {
                    if (v.Weight is { } w && weights.Add(w))
                    {
                        remaining += w.ByteSize;
                    }
                    else if (v.State is { } st && states.Add(st))
                    {
                        remaining += PlacementPlanner.StateBytes(st, options);
                    }
                }
            }

            if (!twinsCompiled)
            {
                // One layer's streamed experts and the sparse banks: at most the largest streamed segment's weights twice.
                remaining += 2 * plan.StreamedSegments.Where(kv => plan.Slots[kv.Value].Device == device)
                    .Select(kv => plan.Segments[kv.Key].Nodes.SelectMany(n => n.Inputs).Where(v => v.Weight is not null).Select(v => v.Weight!).Distinct().Sum(w => w.ByteSize))
                    .DefaultIfEmpty(0).Max();
            }

            worst = Math.Max(worst, remaining - device.GetMemoryInfo().Available);
        }

        return worst;
    }

    /// <summary>
    /// Places the graph (its dynamic bounds already added) and compiles every segment on its device: the largest segment of each accelerator
    /// first (<see cref="CompileOrder"/>), then the streamed compilations, then the rest. With <paramref name="retryOnStreamingOom"/> a streamed
    /// compilation that runs out of memory fails the compilation (the caller places the graph again with more memory kept free) instead of
    /// leaving its segment to the CPU for prefill.
    /// </summary>
    /// <param name="graph">The graph to compile.</param>
    /// <param name="placement">Devices, fallback, memory reserve and compile options.</param>
    /// <param name="retryOnStreamingOom">Whether a streamed compilation that runs out of memory fails the compilation.</param>
    /// <param name="shortfall">Set, when a device runs out of memory, to what the rest of the plan lacked (<see cref="MemoryShortfall"/>).</param>
    private static CompiledModel CompilePlaced(ComputeGraph graph, PlacementOptions placement, bool retryOnStreamingOom, StrongBox<long> shortfall)
    {
        var ct = placement.Compile.CancellationToken;
        ct.ThrowIfCancellationRequested();
        var placing = new LoadProgressCounter(placement.Compile.Progress, LoadStage.Placing, 1, graph.Name);
        var plan = PlacementPlanner.Plan(graph, placement);
        placing.Complete();
        // Segments report through one combined count per stage.
        var progress = placement.Compile.Progress is { } target ? new SegmentProgress(target) : null;
        var segmentOptions = (progress is null ? placement.Compile : placement.Compile.WithProgress(progress)).WithStateScope(new object());
        var compiled = new ICompiledSegment[plan.Segments.Count];
        int buffers = Math.Clamp(placement.StreamBuffers, 1, 2);
        var streamed = new ICompiledSegment?[plan.Segments.Count];
        var sparse = new ICompiledSegment?[plan.Segments.Count];
        var order = CompileOrder(plan);
        bool twinsCompiled = false;
        try
        {
            for (int i = 0; i < order.Indices.Count; i++)
            {
                if (i == order.FirstCount)
                {
                    CompileStreamed(plan, segmentOptions, buffers, retryOnStreamingOom, streamed, ct);
                    CompileSparse(plan, segmentOptions, sparse, ct);
                    twinsCompiled = true;
                }

                ct.ThrowIfCancellationRequested();
                int index = order.Indices[i];
                var seg = plan.Segments[index];
                compiled[index] = seg.Nodes switch
                {
                    [{ Op.Name: "Loop", Op.Category: OpCategory.ControlFlow }] => new LoopSegment(seg, placement),
                    [{ Op.Category: OpCategory.ControlFlow }] => new IfSegment(seg, placement),
                    _ => plan.Slots[seg.DeviceSlot].Device.Compile(seg, segmentOptions),
                };
                progress?.NextSegment();
            }
        }
        catch (Exception e)
        {
            if (e is DeviceOutOfMemoryException)
            {
                shortfall.Value = MemoryShortfall(plan, placement.Compile, compiled, twinsCompiled);
            }

            foreach (var c in compiled.Concat(streamed).Concat(sparse))
            {
                c?.Dispose();
            }

            throw;
        }

        if (Environment.GetEnvironmentVariable("SHARPLIGHT_LOAD_TIMING") is "1" or "true")
        {
            foreach (var slot in plan.Slots.Where(s => s.Device.Info.Kind != DeviceKind.Cpu).Distinct())
            {
                var memory = slot.Device.GetMemoryInfo();
                Console.Error.WriteLine($"[load] {slot.Device.Info.Name} after compiling '{graph.Name}': {memory.Available >> 20} MiB available, "
                    + $"{memory.UsedByUs >> 20} MiB held, {plan.Segments.Count(s => plan.Slots[s.DeviceSlot] == slot)} segments, {streamed.Count(s => s is not null)} streamed, "
                    + $"{sparse.Count(s => s is not null)} sparse, "
                    + $"{placement.ReservePerDevice >> 20} MiB reserved");
            }

            if (placement.Fallback is { } fallback)
            {
                int fallbackSlot = plan.Slots.Count - 1;
                Console.Error.WriteLine($"[load] groups on {fallback.Device.Info.Name}: " + string.Join(", ", plan.GroupDevice.Where(kv => kv.Value == fallbackSlot).Select(kv => kv.Key)));
            }
        }

        return new CompiledModel(graph, plan, placement.Compile, compiled, streamed, placement.StreamExpertsMinRows, buffers, sparse);
    }

    /// <summary>
    /// The order in which the segments are compiled: on every accelerator the segment with the most nodes first, then the others in graph order.
    /// The segments of a device share one activation arena (<see cref="SharedArena{T}"/>), which a segment that needs more than the arena holds
    /// replaces for itself and the segments after it while the earlier ones keep theirs: compiling the largest segment first (normally the
    /// one with the most nodes, e.g. the layers between routed experts on the CPU) lets the smaller ones share its arena instead of each size
    /// step holding its own (Qwen3.5-122B with the experts of 10 layers on the CPU: 166 + 174 + 274 MiB at 512 rows, now 274).
    /// </summary>
    /// <returns>The segment indices in compile order and how many of them come first (the largest of each accelerator).</returns>
    private static (List<int> Indices, int FirstCount) CompileOrder(PlacementPlan plan)
    {
        var first = plan.Segments.Select((s, i) => (Segment: s, Index: i))
            .Where(x => plan.Slots[x.Segment.DeviceSlot].Device.Info.Kind != DeviceKind.Cpu && x.Segment.Nodes is not [{ Op.Category: OpCategory.ControlFlow }])
            .GroupBy(x => x.Segment.DeviceSlot)
            .Select(g => g.MaxBy(x => x.Segment.Nodes.Count).Index)
            .ToHashSet();
        List<int> indices = [.. first.Order(), .. Enumerable.Range(0, plan.Segments.Count).Where(i => !first.Contains(i))];
        return (indices, first.Count);
    }

    /// <summary>
    /// Compiles, for the streamed segments (<see cref="PlacementPlan.StreamedSegments"/>) whose expert products read one ids input and whose
    /// outputs are sums over the slots, a GPU version for passes of up to <see cref="SparseExpertRows"/> rows that copies only the selected
    /// experts in (<see cref="CompileOptions.SparseExpertSlots"/>) into <paramref name="sparse"/>. Devices that cannot leave the segment out.
    /// </summary>
    private static void CompileSparse(PlacementPlan plan, CompileOptions segmentOptions, ICompiledSegment?[] sparse, CancellationToken ct)
    {
        if (SparseExpertRows <= 0)
        {
            return;
        }

        foreach (var (index, slot) in plan.StreamedSegments.OrderBy(kv => kv.Key))
        {
            var segment = plan.Segments[index];
            if (!plan.Slots[slot].Device.StreamsSelectedExperts || SparseExpertIds(segment) is not { } ids || !SumsOverSlots(segment)
                || !SlotWeights(segment, ids).Qualifies || segment.Nodes.Any(n => n.Op.Name == "MatMulId" && n.Inputs[1].Shape[0].Fixed / SparseMinExpertsPerSlot < SparseExpertRows * ids.Shape[1].Fixed))
            {
                continue;
            }

            ct.ThrowIfCancellationRequested();
            var twin = new GraphSegment { Graph = segment.Graph, Index = segment.Index, Nodes = segment.Nodes, Inputs = segment.Inputs, Outputs = segment.Outputs, DeviceSlot = slot };
            try
            {
                var compiled = plan.Slots[slot].Device.Compile(twin, segmentOptions.WithSparseExperts(SparseExpertRows, SparseExpertRows * (int)ids.Shape[1].Fixed));
                if (compiled is IStreamedSegment { CopiesAsynchronously: false })
                {
                    // Its copies would take the host thread before the CPU's share starts: the split would be slower than the CPU alone.
                    compiled.Dispose();
                    throw new NotSupportedException("the expert banks are not page-locked");
                }

                sparse[index] = compiled;
            }
            catch (Exception e) when (e is DeviceOutOfMemoryException or InsufficientMemoryException or NotSupportedException)
            {
                if (Environment.GetEnvironmentVariable("SHARPLIGHT_LOAD_TIMING") is "1" or "true")
                {
                    Console.Error.WriteLine($"[load] segment {index} has no sparse expert version on {plan.Slots[slot].Device.Info.Name}: {e.Message}");
                }
            }
        }
    }

    /// <summary>The ids input that every expert product (<c>MatMulId</c>) of the segment reads, when there is exactly one with a fixed slot count.</summary>
    private static Value? SparseExpertIds(GraphSegment segment)
    {
        var ids = segment.Nodes.Where(n => n.Op.Name == "MatMulId").Select(n => n.Inputs[2]).Distinct().ToList();
        return ids is [{ Shape.Rank: 2 } only] && !only.Shape[1].IsSymbolic && segment.Inputs.Contains(only) ? only : null;
    }

    /// <summary>
    /// Whether a split pass of the segment gives the sum of its two shares, and the slot weights to zero for that. Slots without an expert
    /// (id -1) give zero rows in expert products; when the segment also reads the ids elsewhere (rows of per-expert tables such as biases),
    /// such a slot has a value, and the pass qualifies only when one input of slot weights ([rows, k] or [rows, k, 1], f32) multiplies the
    /// slots right before a <c>ReduceSum</c> (possibly behind views): zero weights then drop those slots.
    /// </summary>
    private static (bool Qualifies, Value? Weights) SlotWeights(GraphSegment segment, Value ids)
    {
        if (segment.Nodes.All(n => !n.Inputs.Contains(ids) || n.Op.Name == "MatMulId" && n.Inputs[2] == ids))
        {
            return (true, null);
        }

        long k = ids.Shape[1].Fixed;
        var candidates = segment.Inputs.Where(v => v != ids && v.Type == Tensors.DataType.F32 && v.Shape.Rank is 2 or 3 && v.Shape[1] is { IsSymbolic: false } slots
            && slots.Fixed == k && (v.Shape.Rank == 2 || v.Shape[2] is { IsSymbolic: false, Fixed: 1 }) && ScalesSlotsBeforeSum(segment, v)).ToList();
        return candidates is [var weights] ? (true, weights) : (false, null);
    }

    /// <summary>Whether <paramref name="weights"/> reaches, through views, a <c>Mul</c> whose product reaches a <c>ReduceSum</c> through views.</summary>
    private static bool ScalesSlotsBeforeSum(GraphSegment segment, Value weights)
    {
        foreach (var mul in Consumers(segment, weights).Where(n => n.Op.Name == "Mul"))
        {
            if (Consumers(segment, mul.Outputs[0]).Any(n => n.Op.Name == "ReduceSum"))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>The nodes of the segment that read <paramref name="value"/>, directly or through views (the views themselves excluded).</summary>
    private static IEnumerable<Node> Consumers(GraphSegment segment, Value value)
    {
        foreach (var n in segment.Nodes.Where(n => n.Inputs.Contains(value)))
        {
            if (n.Op.Category != OpCategory.View)
            {
                yield return n;
                continue;
            }

            foreach (var behind in Consumers(segment, n.Outputs[0]))
            {
                yield return behind;
            }
        }
    }

    /// <summary>
    /// Whether every output of the segment is an f32 sum over the slot axis (a <c>ReduceSum</c>, possibly behind views), so that the sums of
    /// two passes over disjoint slots add up to the pass over all.
    /// </summary>
    private static bool SumsOverSlots(GraphSegment segment)
    {
        foreach (var output in segment.Outputs)
        {
            var producer = output.Producer;
            while (producer is { Op.Category: OpCategory.View } && segment.Nodes.Contains(producer))
            {
                producer = producer.Inputs[0].Producer;
            }

            if (output.Type != Tensors.DataType.F32 || producer is not { Op.Name: "ReduceSum" } || !segment.Nodes.Contains(producer))
            {
                return false;
            }
        }

        return segment.Outputs.Count > 0;
    }

    /// <summary>
    /// Runs segment <paramref name="i"/> split between its CPU compilation and its sparse GPU compilation when the pass is small enough: the
    /// GPU takes the first slots of every row (it copies in the experts they select), the CPU the others (ids of -1 select no expert), both
    /// at the same time; the outputs, sums over the slots, are added. The GPU's share per segment and row count comes from a
    /// <see cref="SplitBalance"/> fed with the pass times (0: the CPU alone).
    /// </summary>
    /// <returns>False, running nothing, when the segment has no sparse compilation or the pass does not qualify.</returns>
    private bool RunSplit(int i, RunParameters parameters)
    {
        if (_sparse[i] is not { } gpu || _sparseIds[i] is not { } idsValue || _residentIn[_streamBufferOf[i]] == i
            || !_hostValues.TryGetValue(idsValue, out var hostIds))
        {
            return false;
        }

        var cpu = _segments[i];
        int k = (int)idsValue.Shape[1].Fixed;
        long rows = idsValue.Shape.Resolve(parameters.Symbols)[0];
        if (rows > SparseExpertRows || rows <= 0)
        {
            return false;
        }

        int bytes = (int)ByteCount(idsValue, parameters);
        long outBytes = cpu.Segment.Outputs.Where(o => IsConsumedAfter(i, o)).Select(o => ByteCount(o, parameters)).DefaultIfEmpty(0).Max();
        var weightsValue = _slotWeights[i];
        byte[]? hostWeights = null;
        if (weightsValue is not null && (!_hostValues.TryGetValue(weightsValue, out hostWeights) || ByteCount(weightsValue, parameters) != bytes))
        {
            return false;
        }

        if (!_splitBuffers.TryGetValue(i, out var buffers) || buffers.GpuIds.Length < bytes || buffers.Out.Length < outBytes)
        {
            _splitBuffers[i] = buffers = new SplitBuffers(bytes, outBytes);
        }

        long passStart = Stopwatch.GetTimestamp();
        if (!_balances.TryGetValue((i, rows), out var balance))
        {
            _balances[(i, rows)] = balance = new SplitBalance(k, k / 2);
        }

        int kg = Math.Clamp(DiagnosticSparseGpuSlots ?? balance.Current, 0, k);
        var all = MemoryMarshal.Cast<byte, int>(hostIds.AsSpan(0, bytes));
        var gpuIds = MemoryMarshal.Cast<byte, int>(buffers.GpuIds.AsSpan(0, bytes));
        var cpuIds = MemoryMarshal.Cast<byte, int>(buffers.CpuIds.AsSpan(0, bytes));
        for (int j = 0; j < all.Length; j++)
        {
            bool onGpu = j % k < kg;
            gpuIds[j] = onGpu ? all[j] : -1;
            cpuIds[j] = onGpu ? -1 : all[j];
        }

        if (hostWeights is not null)
        {
            var weights = MemoryMarshal.Cast<byte, float>(hostWeights.AsSpan(0, bytes));
            var gpuWeights = MemoryMarshal.Cast<byte, float>(buffers.GpuWeights.AsSpan(0, bytes));
            var cpuWeights = MemoryMarshal.Cast<byte, float>(buffers.CpuWeights.AsSpan(0, bytes));
            for (int j = 0; j < weights.Length; j++)
            {
                bool onGpu = j % k < kg;
                gpuWeights[j] = onGpu ? weights[j] : 0f;
                cpuWeights[j] = onGpu ? 0f : weights[j];
            }
        }

        foreach (var v in cpu.Segment.Inputs)
        {
            // With no GPU slots the pass runs on the CPU alone (the inputs as they are).
            if (v == idsValue && kg > 0)
            {
                gpu.SetInput(v, buffers.GpuIds.AsSpan(0, bytes));
                cpu.SetInput(v, buffers.CpuIds.AsSpan(0, bytes));
            }
            else if (v == weightsValue && kg > 0)
            {
                gpu.SetInput(v, buffers.GpuWeights.AsSpan(0, bytes));
                cpu.SetInput(v, buffers.CpuWeights.AsSpan(0, bytes));
            }
            else if ((v.Kind == ValueKind.Intermediate || v == idsValue || v == weightsValue) && _hostValues.TryGetValue(v, out var input))
            {
                var data = input.AsSpan(0, (int)ByteCount(v, parameters));
                if (kg > 0)
                {
                    gpu.SetInput(v, data);
                }

                cpu.SetInput(v, data);
            }
        }

        if (kg > 0)
        {
            gpu.Execute(parameters);
        }

        long cpuStart = Stopwatch.GetTimestamp();
        cpu.Execute(parameters);
        double cpuMs = Stopwatch.GetElapsedTime(cpuStart).TotalMilliseconds;
        double waitMs = 0;
        foreach (var o in cpu.Segment.Outputs)
        {
            if (!IsConsumedAfter(i, o))
            {
                continue;
            }

            int n = (int)ByteCount(o, parameters);
            if (!_hostValues.TryGetValue(o, out var sum) || sum.Length < n)
            {
                _hostValues[o] = sum = new byte[n];
            }

            cpu.GetOutput(o, sum.AsSpan(0, n));
            if (kg == 0)
            {
                continue;
            }

            long waitStart = Stopwatch.GetTimestamp();
            gpu.GetOutput(o, buffers.Out.AsSpan(0, n));
            waitMs += Stopwatch.GetElapsedTime(waitStart).TotalMilliseconds;
            var total = MemoryMarshal.Cast<byte, float>(sum.AsSpan(0, n));
            var part = MemoryMarshal.Cast<byte, float>(buffers.Out.AsSpan(0, n));
            for (int j = 0; j < total.Length; j++)
            {
                total[j] += part[j];
            }
        }

        if (_splitTimes is not null)
        {
            ref var split = ref _splitTimes[i];
            split = (split.GpuSlots + kg, split.CpuMs + cpuMs, split.WaitMs + waitMs, split.Runs + 1);
        }

        balance.Record(kg, Stopwatch.GetElapsedTime(passStart).TotalMilliseconds);
        return true;
    }

    /// <summary>
    /// Compiles the streamed GPU versions of the segments the plan lists (<see cref="PlacementPlan.StreamedSegments"/>) into
    /// <paramref name="streamed"/>. A segment that does not compile (an unsupported kernel; no memory left unless <paramref name="throwOnOom"/>)
    /// keeps running on the CPU only.
    /// </summary>
    private static void CompileStreamed(PlacementPlan plan, CompileOptions segmentOptions, int buffers, bool throwOnOom, ICompiledSegment?[] streamed, CancellationToken ct)
    {
        int ordinal = 0;
        foreach (var (index, slot) in plan.StreamedSegments.OrderBy(kv => kv.Key))
        {
            ct.ThrowIfCancellationRequested();
            var options = segmentOptions.WithStreamedWeights(ordinal % buffers);
            var segment = plan.Segments[index];
            var twin = new GraphSegment { Graph = segment.Graph, Index = segment.Index, Nodes = segment.Nodes, Inputs = segment.Inputs, Outputs = segment.Outputs, DeviceSlot = slot };
            try
            {
                streamed[index] = plan.Slots[slot].Device.Compile(twin, options);
                ordinal++;
            }
            catch (Exception e) when (e is DeviceOutOfMemoryException && !throwOnOom || e is InsufficientMemoryException or NotSupportedException)
            {
                if (Environment.GetEnvironmentVariable("SHARPLIGHT_LOAD_TIMING") is "1" or "true")
                {
                    Console.Error.WriteLine($"[load] segment {index} is not streamed to {plan.Slots[slot].Device.Info.Name}: {e.Message}");
                }
            }
        }
    }

    /// <summary>
    /// Before segment <paramref name="i"/> runs, starts the copies of the streamed segments after it whose memory is free: in order, until
    /// one needs a memory that the segment running now or a streamed segment before it still uses (<see cref="PlacementOptions.StreamBuffers"/>).
    /// </summary>
    private void BeginStreamingAhead(int i, RunParameters parameters)
    {
        Span<bool> busy = stackalloc bool[_streamBuffers];
        if (_streamed[i] is not null && Runner(i, parameters) == _streamed[i])
        {
            busy[StreamBufferOf(i)] = true;
        }

        for (int j = i + 1; j < _segments.Length; j++)
        {
            if (_streamed[j] is null)
            {
                continue;
            }

            if (Runner(j, parameters) is not IStreamedSegment next || busy[StreamBufferOf(j)])
            {
                return;
            }

            Stream(j, next);
            busy[StreamBufferOf(j)] = true;
        }
    }

    /// <summary>The streamed-weight memory of streamed segment <paramref name="i"/> (alternating in segment order).</summary>
    private int StreamBufferOf(int i) => _streamBufferOf[i];

    /// <summary>
    /// The segment that runs position <paramref name="i"/>: its streamed GPU version when the pass has enough rows or the streamed-weight
    /// memory still holds its weights, else the segment.
    /// </summary>
    private ICompiledSegment Runner(int i, RunParameters parameters)
    {
        if (_streamed[i] is not { } streamed || _streamMinRows <= 0)
        {
            return _segments[i];
        }

        if (_residentIn[_streamBufferOf[i]] == i)
        {
            return streamed;
        }

        var inputs = _segments[i].Segment.Inputs;
        long rows = inputs.Count > 0 && inputs[0].Shape.Rank > 0 ? inputs[0].Shape.Resolve(parameters.Symbols)[0] : 0;
        return rows >= _streamMinRows ? streamed : _segments[i];
    }

    /// <summary>
    /// Adds the upper bounds of the graph's data-dependent symbols (<see cref="ComputeGraph.DynamicSymbols"/>) to the symbol bounds, so providers size
    /// buffers and specialize kernels for them like for any caller-bound symbol. Bounds are evaluated in creation order, so a bound may refer
    /// to earlier dynamic symbols.
    /// </summary>
    private static PlacementOptions WithDynamicBounds(ComputeGraph graph, PlacementOptions placement)
    {
        if (graph.DynamicSymbols.Count == 0)
        {
            return placement;
        }

        var compile = placement.Compile;
        foreach (var (name, bound) in graph.DynamicSymbols)
        {
            long value = bound.IsSymbolic ? compile.Bound(bound.Symbol!) : bound.Fixed;
            compile = compile.WithSymbolBounds([new(name, Math.Max(1, value))]);
        }

        return placement.WithCompile(compile);
    }

    /// <summary>
    /// The actual shape of a graph output after <see cref="Run"/>: symbolic dimensions are evaluated with the caller's bindings and the values
    /// data-dependent ops wrote into <paramref name="parameters"/>.
    /// </summary>
    /// <param name="name">Output name.</param>
    /// <param name="parameters">The parameters passed to the last <see cref="Run"/>.</param>
    public long[] OutputShape(string name, RunParameters parameters) => Graph.Outputs[name].Shape.Resolve(parameters.Symbols);

    /// <summary>Device memory held on accelerators (segments placed on non-CPU devices): what counts against GPU memory budgets.</summary>
    public long AcceleratorMemoryBytes
        => _segments.Where(s => Plan.Slots[s.Segment.DeviceSlot].Device.Info.Kind != DeviceKind.Cpu).Sum(s => s.DeviceMemoryBytes);

    /// <summary>
    /// Memory held by all segments on whatever device they run: weights (a memory-mapped CPU weight counts as resident), activation arenas and
    /// state such as KV caches. Unlike <see cref="AcceleratorMemoryBytes"/> it includes CPU segments, so a model loaded on the CPU reports what it occupies.
    /// </summary>
    public long ResidentMemoryBytes => _segments.Sum(s => s.DeviceMemoryBytes);

    /// <summary>Supplies a graph input for the next <see cref="Run"/> (dense logical layout).</summary>
    public void SetInput(Value input, ReadOnlySpan<byte> data)
    {
        foreach (var s in _segments)
        {
            if (s.Segment.Inputs.Contains(input))
            {
                s.SetInput(input, data);
            }
        }

        foreach (var s in _streamed.Concat(_sparse))
        {
            if (s is not null && s.Segment.Inputs.Contains(input))
            {
                s.SetInput(input, data);
            }
        }
    }

    /// <summary>Typed overload of <see cref="SetInput(Value, ReadOnlySpan{byte})"/>.</summary>
    public void SetInput<T>(Value input, ReadOnlySpan<T> data) where T : unmanaged
        => SetInput(input, MemoryMarshal.AsBytes(data));

    /// <summary>
    /// Supplies a string graph input (<see cref="Tensors.DataType.Str"/>) for the next <see cref="Run"/>, in row-major order. The strings are
    /// added to <see cref="RunParameters.Strings"/> when the run starts.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="input"/> is not a string value.</exception>
    public void SetInput(Value input, ReadOnlySpan<string> strings)
    {
        if (input.Type != Tensors.DataType.Str)
        {
            throw new ArgumentException($"Input '{input.Name}' is {input.Type}, not a string tensor.", nameof(input));
        }

        _stringInputs[input] = strings.ToArray();
    }

    /// <summary>Reads a string graph output (<see cref="Tensors.DataType.Str"/>) after <see cref="Run"/>, in row-major order.</summary>
    /// <exception cref="InvalidOperationException">Called before <see cref="Run"/>.</exception>
    public void GetOutput(string name, Span<string> destination)
    {
        var run = _lastRun ?? throw new InvalidOperationException("GetOutput called before Run.");
        var handles = new int[destination.Length];
        GetOutput(name, MemoryMarshal.AsBytes(handles.AsSpan()));
        for (int i = 0; i < handles.Length; i++)
        {
            destination[i] = run.Strings[handles[i]];
        }
    }

    /// <summary>
    /// Executes all segments in order, forwarding boundary activations through host memory. Not thread-safe: one run at a time per instance.
    /// </summary>
    /// <param name="parameters">
    /// Bindings for symbolic dimensions and runtime scalars; must cover every caller-bound symbol and scalar the graph uses. Data-dependent ops
    /// add the values of their symbols (<see cref="ComputeGraph.DynamicSymbols"/>) to <see cref="RunParameters.Symbols"/>.
    /// </param>
    public void Run(RunParameters parameters)
    {
        _lastRun = parameters;
        parameters.Strings.Clear();
        foreach (var (input, strings) in _stringInputs)
        {
            var handles = new int[strings.Length];
            for (int i = 0; i < strings.Length; i++)
            {
                handles[i] = parameters.Strings.Add(strings[i]);
            }

            SetInput(input, MemoryMarshal.AsBytes(handles.AsSpan()));
        }

        for (int i = 0; i < _segments.Length; i++)
        {
            long started = _segmentTimes is null ? 0 : Stopwatch.GetTimestamp();
            if (RunSplit(i, parameters))
            {
                if (_segmentTimes is not null)
                {
                    ref var split = ref _segmentTimes[i];
                    split = (split.Ticks + Stopwatch.GetTimestamp() - started, split.Runs + 1, true, split.GpuMs);
                }

                continue;
            }

            var seg = Runner(i, parameters);
            if (seg is IStreamedSegment streamed && seg == _streamed[i])
            {
                Stream(i, streamed);
            }

            BeginStreamingAhead(i, parameters);

            // Forward activations produced by earlier segments.
            foreach (var v in seg.Segment.Inputs)
            {
                if (v.Kind == ValueKind.Intermediate && !_forwardedInputs[i].Contains(v) && _hostValues.TryGetValue(v, out var bytes))
                {
                    seg.SetInput(v, bytes.AsSpan(0, (int)ByteCount(v, parameters)));
                }
            }

            var skip = SkippedDownloads(i, parameters);
            if (skip.Count == 0)
            {
                seg.Execute(parameters);
            }
            else
            {
                seg.Execute(parameters, skip);
            }

            // Pull values needed by later segments.
            if (i < _segments.Length - 1)
            {
                foreach (var o in seg.Segment.Outputs)
                {
                    if (!IsConsumedAfter(i, o) || _deviceOnlyOutputs[i].Contains(o))
                    {
                        continue;
                    }

                    var n = ByteCount(o, parameters);
                    if (!_hostValues.TryGetValue(o, out var buf) || buf.Length < n)
                    {
                        _hostValues[o] = buf = new byte[n];
                    }

                    seg.GetOutput(o, buf.AsSpan(0, (int)n));
                }
            }

            if (_segmentTimes is not null)
            {
                ref var time = ref _segmentTimes[i];
                time = (time.Ticks + Stopwatch.GetTimestamp() - started, time.Runs + 1, seg != _segments[i], time.GpuMs + seg.LastRunTimings.GpuMs);
            }
        }
    }

    /// <summary>
    /// With <c>SHARPLIGHT_SEGMENT_TIMING=1</c>, the host time per run of every segment since the last call (inputs, execution and the outputs
    /// later segments read; an asynchronous segment's device time shows where it is waited for), one line per segment; else null. Clears the times.
    /// </summary>
    public string? TakeSegmentTimings()
    {
        if (_segmentTimes is null)
        {
            return null;
        }

        var lines = new System.Text.StringBuilder();
        double total = 0;
        for (int i = 0; i < _segmentTimes.Length; i++)
        {
            var (ticks, runs, streamed, gpuMs) = _segmentTimes[i];
            double ms = runs == 0 ? 0 : ticks * 1000.0 / Stopwatch.Frequency / runs;
            total += ms;
            string device = streamed ? "streamed" : Plan.Slots[_segments[i].Segment.DeviceSlot].Device.Info.Kind.ToString();
            lines.Append(CultureInfo.InvariantCulture, $"  segment {i,3} {device,-8} {ms,8:F3} ms/run ({runs} runs)");
            if (gpuMs > 0)
            {
                lines.Append(CultureInfo.InvariantCulture, $", device {gpuMs / runs:F3} ms");
            }

            if (_splitTimes![i] is { Runs: > 0 } split)
            {
                lines.Append(CultureInfo.InvariantCulture,
                    $", split: {(double)split.GpuSlots / split.Runs:F2} GPU slots, CPU {split.CpuMs / split.Runs:F3} ms, wait for the GPU {split.WaitMs / split.Runs:F3} ms");
            }

            lines.AppendLine();
        }

        lines.Append(CultureInfo.InvariantCulture, $"  total {total:F3} ms/run");
        Array.Clear(_segmentTimes);
        Array.Clear(_splitTimes!);
        return lines.ToString();
    }

    /// <summary>
    /// Prepares every segment's work for a run with these symbol bindings and skipped outputs without running it (see
    /// <see cref="ICompiledSegment.Prepare"/>): providers that record work per shape record it now instead of at the first such run.
    /// Data-dependent symbols are prepared at their compile-time bounds, as runs record them.
    /// </summary>
    /// <param name="parameters">Symbol bindings and <see cref="RunParameters.SkipOutputs"/> of the runs to come; inputs are not read.</param>
    public void Prepare(RunParameters parameters)
    {
        for (int i = 0; i < _segments.Length; i++)
        {
            _segments[i].Prepare(parameters, SkippedDownloads(i, parameters));
        }
    }

    private bool IsConsumedAfter(int segment, Value v) => _segments.Skip(segment + 1).Any(s => s.Segment.Inputs.Contains(v));

    /// <summary>
    /// Outputs of segment <paramref name="segment"/> that are not read back: graph outputs listed in <see cref="RunParameters.SkipOutputs"/>
    /// that no later segment needs, and the outputs that only reach device-forwarding consumers.
    /// </summary>
    private HashSet<Value> SkippedDownloads(int segment, RunParameters parameters)
    {
        if (parameters.SkipOutputs.Count == 0)
        {
            return _deviceOnlyOutputs[segment].Count == 0 ? s_noSkips : _deviceOnlyOutputs[segment];
        }

        var skip = new HashSet<Value>(_deviceOnlyOutputs[segment]);
        foreach (var name in parameters.SkipOutputs)
        {
            if (Graph.Outputs.TryGetValue(name, out var v) && _segments[segment].Segment.Outputs.Contains(v) && !IsConsumedAfter(segment, v))
            {
                skip.Add(v);
            }
        }

        return skip;
    }

    /// <summary>Reads a graph output after <see cref="Run"/>.</summary>
    public void GetOutput(string name, Span<byte> destination)
    {
        var v = Graph.Outputs[name];
        var seg = _segments.Last(s => s.Segment.Outputs.Contains(v));
        seg.GetOutput(v, destination);
    }

    /// <summary>
    /// Reads <c>destination.Length</c> bytes of a graph output starting at <paramref name="byteOffset"/> (dense logical layout)
    /// after <see cref="Run"/>. Several threads may read disjoint ranges at once when the producing segment supports partial
    /// reads (<see cref="IPartialOutputSegment"/>); otherwise the output is read whole and sliced.
    /// </summary>
    /// <exception cref="InvalidOperationException">Called before <see cref="Run"/>.</exception>
    public void GetOutput(string name, long byteOffset, Span<byte> destination)
    {
        var v = Graph.Outputs[name];
        var seg = _segments.Last(s => s.Segment.Outputs.Contains(v));
        if (seg is IPartialOutputSegment partial)
        {
            partial.GetOutput(v, byteOffset, destination);
            return;
        }

        var run = _lastRun ?? throw new InvalidOperationException("GetOutput called before Run.");
        var whole = new byte[ByteCount(v, run)];
        seg.GetOutput(v, whole);
        whole.AsSpan(checked((int)byteOffset), destination.Length).CopyTo(destination);
    }

    /// <summary>Typed overload of <see cref="GetOutput(string, Span{byte})"/>.</summary>
    public void GetOutput<T>(string name, Span<T> destination) where T : unmanaged
        => GetOutput(name, MemoryMarshal.AsBytes(destination));

    /// <summary>
    /// Whether <see cref="Run"/> can return before the device finishes: the last segment executes asynchronously
    /// (<see cref="ICompiledSegment.ExecutesAsynchronously"/>), so work between <see cref="Run"/> and the first <see cref="GetOutput(string, Span{byte})"/> overlaps it.
    /// </summary>
    public bool RunsAsynchronously => _segments.Length > 0 && _segments[^1].ExecutesAsynchronously;

    /// <summary>Where the most recent <see cref="Run"/> spent its time, summed over the segments (diagnostics).</summary>
    public SegmentRunTimings LastRunTimings
    {
        get
        {
            var sum = default(SegmentRunTimings);
            foreach (var s in _segments)
            {
                sum += s.LastRunTimings;
            }

            return sum;
        }
    }

    /// <summary>Clears the persistent state (KV caches, recurrent state) of every segment, starting a new sequence.</summary>
    public void ResetState()
    {
        foreach (var s in _segments)
        {
            s.ResetState();
        }
    }

    /// <summary>Whether every segment can copy rows of its persistent state (<see cref="CopyStateRows"/>).</summary>
    public bool CanCopyStateRows => _segments.All(s => s.CanCopyStateRows);

    /// <summary>
    /// Copies <paramref name="count"/> rows of the outermost dimension of the named states, from row <paramref name="from"/> on, to
    /// the rows from <paramref name="to"/> on, in order with <see cref="Run"/> (e.g. a sequence slot's recurrent state into a spare row of
    /// a multi-sequence graph and back, or one paged KV block's rows into another). Returns false, copying nothing, when a segment cannot
    /// copy state.
    /// </summary>
    /// <param name="states">State names (<see cref="Models.StateDefinition.Name"/>); each must have a fixed outermost dimension.</param>
    /// <param name="from">First source row.</param>
    /// <param name="to">First destination row.</param>
    /// <param name="count">Rows to copy (the source and destination ranges must not overlap).</param>
    /// <exception cref="ArgumentException">The ranges overlap.</exception>
    public bool CopyStateRows(IReadOnlyCollection<string> states, int from, int to, int count = 1)
    {
        if (!CanCopyStateRows)
        {
            return false;
        }

        if (from == to || states.Count == 0 || count <= 0)
        {
            return true;
        }

        if (Math.Abs((long)from - to) < count)
        {
            throw new ArgumentException("The source and destination rows overlap.", nameof(count));
        }

        foreach (var s in _segments)
        {
            s.CopyStateRows(states, from, to, count);
        }

        return true;
    }

    internal static long ByteCount(Value v, RunParameters p) => v.Type.ByteSize(v.Shape.Elements(p.Symbols));

    /// <summary>Releases the compiled segments and their device memory.</summary>
    public void Dispose()
    {
        foreach (var s in _segments)
        {
            s.Dispose();
        }

        foreach (var s in _streamed.Concat(_sparse))
        {
            s?.Dispose();
        }
    }
}
