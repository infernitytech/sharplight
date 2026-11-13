using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Runtime;

/// <summary>A contiguous run of graph nodes placed on one device.</summary>
public sealed class GraphSegment
{
    public required ComputeGraph Graph { get; init; }
    public required int Index { get; init; }
    public required IReadOnlyList<Node> Nodes { get; init; }
    /// <summary>Values consumed by this segment but produced outside of it (graph inputs or earlier segments).</summary>
    public required IReadOnlyList<Value> Inputs { get; init; }
    /// <summary>Values produced here and needed by later segments or returned as graph outputs.</summary>
    public required IReadOnlyList<Value> Outputs { get; init; }
    public required int DeviceSlot { get; init; }

    public IEnumerable<Value> Weights => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.Weight).Distinct();
    public IEnumerable<Value> States => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.State).Distinct();
}

/// <summary>A device participating in a placement, with its share of the model.</summary>
public sealed class DeviceSlot
{
    public required IComputeDevice Device { get; init; }
    /// <summary>Relative share of layers (like llama.cpp's tensor split). Null = proportional to free memory.</summary>
    public double? Weight { get; init; }
    /// <summary>Hard memory cap for this placement (null = device available memory).</summary>
    public long? MemoryLimit { get; init; }
}

public sealed class PlacementOptions
{
    /// <summary>Devices in pipeline order (first device runs the first layers).</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device for unsupported ops and layers that do not fit (CPU offload). Optional.</summary>
    public DeviceSlot? Fallback { get; init; }
    /// <summary>Bytes reserved on every device for activations, scratch and driver overhead.</summary>
    public long ReservePerDevice { get; init; } = 1L << 30;
    public required CompileOptions Compile { get; init; }
}

/// <summary>Estimated device memory of a compiled model, in bytes.</summary>
public readonly record struct ModelFootprint(long WeightBytes, long StateBytes, long ActivationBytes)
{
    public long Total => WeightBytes + StateBytes + ActivationBytes;
    public override string ToString() => $"{Total / 1048576.0:F0} MiB (weights {WeightBytes / 1048576.0:F0}, state {StateBytes / 1048576.0:F0}, activations ~{ActivationBytes / 1048576.0:F0})";
}

/// <summary>Result of placement: which device each group/node goes to and the resulting segments.</summary>
public sealed class PlacementPlan
{
    public required IReadOnlyList<DeviceSlot> Slots { get; init; }
    public required IReadOnlyList<GraphSegment> Segments { get; init; }
    public required IReadOnlyDictionary<string, int> GroupDevice { get; init; }
    public required IReadOnlyDictionary<int, long> BytesPerSlot { get; init; }

    public string Describe()
    {
        var lines = new List<string>();
        for (int s = 0; s < Slots.Count; s++)
        {
            var groups = GroupDevice.Where(kv => kv.Value == s).Select(kv => kv.Key).ToList();
            lines.Add($"slot {s} {Slots[s].Device.Info.Name}: {groups.Count} groups, {BytesPerSlot.GetValueOrDefault(s) / (1024.0 * 1024 * 1024):F2} GiB weights+state");
        }
        lines.Add($"{Segments.Count} segment(s): " + string.Join(" → ", Segments.Select(sg => $"[{sg.Index}@{sg.DeviceSlot}: {sg.Nodes.Count} nodes]")));
        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Pipeline-style partitioner. Placement groups (normally one per layer) are assigned to devices in
/// graph order, filling each device up to its share of the total memory. Groups that fit nowhere
/// go to the fallback device (CPU offload); so do individual nodes the chosen device can't run.
/// </summary>
public static class PlacementPlanner
{
    public static PlacementPlan Plan(ComputeGraph graph, PlacementOptions options)
    {
        var slots = options.Devices.ToList();
        int fallbackSlot = -1;
        if (options.Fallback is not null)
        {
            slots.Add(options.Fallback);
            fallbackSlot = slots.Count - 1;
        }
        if (slots.Count == 0) throw new ArgumentException("No devices given.");

        // ---- cost per group -------------------------------------------------------------------
        var groupOrder = new List<string>();
        var groupBytes = new Dictionary<string, long>();
        var seenWeights = new HashSet<WeightInfo>();
        var seenStates = new HashSet<StateDefinition>();
        foreach (var n in graph.Nodes)
        {
            if (!groupBytes.ContainsKey(n.Group)) { groupBytes[n.Group] = 0; groupOrder.Add(n.Group); }
            foreach (var v in n.Inputs)
            {
                if (v.Weight is { } w && seenWeights.Add(w)) groupBytes[n.Group] += w.ByteSize;
                if (v.State is { } s && seenStates.Add(s)) groupBytes[n.Group] += StateBytes(s, options.Compile);
            }
        }
        long total = groupBytes.Values.Sum();

        // ---- capacity per pipeline device -----------------------------------------------------
        int nPipe = options.Devices.Count;
        var capacity = new long[nPipe];
        var share = new double[nPipe];
        for (int i = 0; i < nPipe; i++)
        {
            var d = options.Devices[i];
            capacity[i] = Math.Max(0, (d.MemoryLimit ?? d.Device.AvailableMemory) - options.ReservePerDevice);
            share[i] = d.Weight ?? capacity[i];
        }
        double shareSum = share.Sum();
        var target = new long[nPipe];
        for (int i = 0; i < nPipe; i++)
            target[i] = shareSum > 0 ? (long)Math.Ceiling(total * (share[i] / shareSum)) : 0;

        // ---- greedy fill in order -------------------------------------------------------------
        var groupDevice = new Dictionary<string, int>();
        var used = new long[slots.Count];
        int cur = 0;
        foreach (var g in groupOrder)
        {
            long b = groupBytes[g];
            while (cur < nPipe && (used[cur] + b > capacity[cur] || (used[cur] > 0 && used[cur] + b / 2 > target[cur] && cur < nPipe - 1)))
                cur++;
            int slot;
            if (cur < nPipe) slot = cur;
            else if (fallbackSlot >= 0) slot = fallbackSlot;
            else throw new InsufficientMemoryException(
                $"Model does not fit: needs {total / (1024.0 * 1024 * 1024):F2} GiB, group '{g}' ({b / (1024.0 * 1024 * 1024):F2} GiB) has no room and no fallback device is configured.");
            groupDevice[g] = slot;
            used[slot] += b;
        }

        // ---- per-node device (op support fallback) -------------------------------------------
        var nodeSlot = new int[graph.Nodes.Count];
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            int s = groupDevice[n.Group];
            if (n.Op.Category != OpCategory.View && !slots[s].Device.Supports(n).IsSupported)
            {
                if (fallbackSlot >= 0 && slots[fallbackSlot].Device.Supports(n).IsSupported) s = fallbackSlot;
                else throw new NotSupportedException($"Op {n.Op.Name} (node {n.Id}) is not supported on {slots[s].Device.Info.Name}: {slots[s].Device.Supports(n).Reason}");
            }
            nodeSlot[i] = s;
        }
        // View ops follow their input's device to avoid needless transfers.
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            if (n.Op.Category == OpCategory.View && n.Inputs[0].Producer is { } p)
                nodeSlot[i] = nodeSlot[IndexOf(graph, p)];
        }

        var segments = BuildSegments(graph, nodeSlot);
        return new PlacementPlan
        {
            Slots = slots,
            Segments = segments,
            GroupDevice = groupDevice,
            BytesPerSlot = Enumerable.Range(0, slots.Count).ToDictionary(i => i, i => used[i]),
        };
    }

    static int IndexOf(ComputeGraph g, Node n)
    {
        // Node ids are assigned sequentially by the builder, but be robust.
        if (n.Id < g.Nodes.Count && g.Nodes[n.Id] == n) return n.Id;
        for (int i = 0; i < g.Nodes.Count; i++) if (g.Nodes[i] == n) return i;
        throw new InvalidOperationException("Node not in graph.");
    }

    /// <summary>
    /// Device memory a graph needs when compiled with <paramref name="options"/>, before any device is touched: weights, persistent
    /// state (KV caches / recurrent state) and a rough bound for activations and scratch. <paramref name="slots"/> &gt; 1 sizes
    /// the state of a multi-sequence compilation of the single-sequence <paramref name="graph"/> (KV rows for slots · context,
    /// recurrent state per slot). Use it to decide whether a model fits before compiling it.
    /// </summary>
    public static ModelFootprint Estimate(ComputeGraph graph, CompileOptions options, int slots = 1)
    {
        long weights = 0, states = 0, maxOut = 0, maxNode = 0;
        var seenW = new HashSet<WeightInfo>();
        var seenS = new HashSet<StateDefinition>();
        foreach (var n in graph.Nodes)
        {
            long nodeBytes = 0;
            foreach (var v in n.Inputs)
            {
                if (v.Weight is { } w && seenW.Add(w)) weights += w.ByteSize;
                if (v.State is { } st && seenS.Add(st))
                {
                    if (slots > 1 && MultiSequence.IsKvState(st))
                        states += StateBytes(st, new CompileOptions { MaxTokens = options.MaxTokens, ContextLength = options.ContextLength * slots, SymbolBounds = options.SymbolBounds });
                    else states += StateBytes(st, options) * Math.Max(1, slots);
                }
            }
            if (n.Op.Category == OpCategory.View) continue;
            foreach (var o in n.Outputs)
            {
                long b = ValueBytes(o, options);
                nodeBytes += b;
                maxOut = Math.Max(maxOut, b);
            }
            maxNode = Math.Max(maxNode, nodeBytes);
        }
        // Activations live in a liveness-packed arena plus staging copies of the inputs/outputs; a few times the largest node
        // output (matmul in/out, residual, norm) covers the arena, the 256 MiB sub-allocation granule of device memory the rest.
        long act = 4 * maxOut + (256L << 20);
        return new ModelFootprint(weights, states, act);
    }

    static long ValueBytes(Value v, CompileOptions o)
    {
        long elems = 1;
        foreach (var d in v.Shape.Dims) elems *= d.IsSymbolic ? Math.Max(1, d.Symbol is WellKnown.TokenSymbol or WellKnown.ContextSymbol || o.SymbolBounds.ContainsKey(d.Symbol!) ? o.Bound(d.Symbol!) : o.MaxTokens) : d.Fixed;
        return v.Type.ByteSize(elems);
    }

    public static long StateBytes(StateDefinition s, CompileOptions o)
    {
        long elems = 1;
        foreach (var d in s.Shape.Dims) elems *= d.IsSymbolic ? o.Bound(d.Symbol!) : d.Fixed;
        return s.Type.ByteSize(elems);
    }

    static List<GraphSegment> BuildSegments(ComputeGraph graph, int[] nodeSlot)
    {
        var runs = new List<(int Slot, List<Node> Nodes)>();
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            if (runs.Count == 0 || runs[^1].Slot != nodeSlot[i]) runs.Add((nodeSlot[i], []));
            runs[^1].Nodes.Add(graph.Nodes[i]);
        }

        var producerRun = new Dictionary<Value, int>();
        for (int r = 0; r < runs.Count; r++)
            foreach (var n in runs[r].Nodes)
                foreach (var o in n.Outputs) producerRun[o] = r;

        var graphOutputs = graph.Outputs.Values.ToHashSet();
        var segs = new List<GraphSegment>();
        for (int r = 0; r < runs.Count; r++)
        {
            var inputs = new List<Value>();
            var seenIn = new HashSet<Value>();
            foreach (var n in runs[r].Nodes)
                foreach (var v in n.Inputs)
                    if ((v.Kind == ValueKind.Input || (v.Kind == ValueKind.Intermediate && producerRun[v] != r)) && seenIn.Add(v))
                        inputs.Add(v);

            var outputs = new List<Value>();
            foreach (var n in runs[r].Nodes)
                foreach (var o in n.Outputs)
                {
                    bool usedLater = graphOutputs.Contains(o);
                    for (int r2 = r + 1; r2 < runs.Count && !usedLater; r2++)
                        usedLater = runs[r2].Nodes.Any(n2 => n2.Inputs.Contains(o));
                    if (usedLater) outputs.Add(o);
                }

            segs.Add(new GraphSegment
            {
                Graph = graph,
                Index = r,
                Nodes = runs[r].Nodes,
                Inputs = inputs,
                Outputs = outputs,
                DeviceSlot = runs[r].Slot,
            });
        }
        return segs;
    }
}
