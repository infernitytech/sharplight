using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Pipeline-style partitioner. Placement groups (normally one per layer) are assigned to devices in
/// graph order, filling each device up to its share of the total memory. Groups that fit nowhere
/// go to the fallback device (CPU offload); so do individual nodes the chosen device can't run.
/// With a fallback device, the gathers of a lookup table (a weight that only <c>Gather</c> reads: token embeddings, n-gram tables)
/// form a placement group of their own (<see cref="TableGroupSuffix"/>), and a table that does not fit the current device goes to the
/// fallback without taking the following layers with it.
/// </summary>
public static class PlacementPlanner
{
    /// <summary>Suffix of the placement group that holds the gathers of a lookup table, appended to the group of the gathers.</summary>
    public const string TableGroupSuffix = "#table";

    /// <summary>
    /// Distributes a graph over devices and cuts it into segments.
    /// </summary>
    /// <param name="graph">The graph to place.</param>
    /// <param name="options">Devices, fallback and memory reserve.</param>
    /// <exception cref="ArgumentException">No device was given.</exception>
    /// <exception cref="InsufficientMemoryException">A placement group fits on no device and no fallback device is configured.</exception>
    /// <exception cref="NotSupportedException">A node is supported neither by its device nor by the fallback device.</exception>
    public static PlacementPlan Plan(ComputeGraph graph, PlacementOptions options)
    {
        var slots = options.Devices.ToList();
        int fallbackSlot = -1;
        if (options.Fallback is not null)
        {
            slots.Add(options.Fallback);
            fallbackSlot = slots.Count - 1;
        }

        if (slots.Count == 0)
        {
            throw new ArgumentException("No devices given.");
        }

        var groupOf = GroupKeys(graph, fallbackSlot >= 0);
        var (groupOrder, groupBytes) = MeasureGroups(graph, options.Compile, groupOf);
        var groupDevice = AssignGroups(groupOrder, groupBytes, options, slots.Count, fallbackSlot, out long[] used);
        var nodeSlot = AssignNodes(graph, slots, groupDevice, fallbackSlot, groupOf);
        var segments = BuildSegments(graph, nodeSlot, slots, options.GroupDataDependentCounts);
        return new PlacementPlan
        {
            Slots = slots,
            Segments = segments,
            GroupDevice = groupDevice,
            BytesPerSlot = Enumerable.Range(0, slots.Count).ToDictionary(i => i, i => used[i]),
        };
    }

    /// <summary>
    /// The placement group of every node: its own group, or with a fallback device, for the gathers of a lookup table (a weight that only
    /// <c>Gather</c> reads, as its table input) that group plus <see cref="TableGroupSuffix"/>.
    /// </summary>
    private static Func<Node, string> GroupKeys(ComputeGraph graph, bool hasFallback)
    {
        if (!hasFallback)
        {
            return static n => n.Group;
        }

        var gatherOnly = new Dictionary<WeightInfo, bool>();
        foreach (var n in graph.Nodes)
        {
            for (int i = 0; i < n.Inputs.Length; i++)
            {
                if (n.Inputs[i].Weight is { } w)
                {
                    bool gathered = IsTableGather(n) && i == 0;
                    gatherOnly[w] = gatherOnly.TryGetValue(w, out bool before) ? before && gathered : gathered;
                }
            }
        }

        var tables = gatherOnly.Where(kv => kv.Value).Select(kv => kv.Key).ToHashSet();
        return n => IsTableGather(n) && n.Inputs[0].Weight is { } table && tables.Contains(table) ? n.Group + TableGroupSuffix : n.Group;
    }

    private static bool IsTableGather(Node n) => n.Op.Name == "Gather";

    /// <summary>Bytes of weights and state per placement group (each weight and state counted once, in its first group), groups in graph order.</summary>
    private static (List<string> Order, Dictionary<string, long> Bytes) MeasureGroups(ComputeGraph graph, CompileOptions compile, Func<Node, string> groupOf)
    {
        var order = new List<string>();
        var bytes = new Dictionary<string, long>();
        var seenWeights = new HashSet<WeightInfo>();
        var seenStates = new HashSet<StateDefinition>();
        foreach (var n in graph.Nodes)
        {
            string group = groupOf(n);
            if (!bytes.ContainsKey(group))
            {
                bytes[group] = 0;
                order.Add(group);
            }

            foreach (var v in n.Inputs)
            {
                // A weight another compiled model already holds (shared weights) takes no new memory.
                if (v.Weight is { } w && seenWeights.Add(w) && compile.SharedWeights?.IsResident(w) != true)
                {
                    bytes[group] += w.ByteSize;
                }

                if (v.State is { } s && seenStates.Add(s))
                {
                    bytes[group] += StateBytes(s, compile);
                }
            }
        }

        return (order, bytes);
    }

    /// <summary>
    /// Greedy fill in graph order: a group moves to the next pipeline device when it does not fit the current one, or when the
    /// current one already holds its share. Groups that fit nowhere go to the fallback slot. A lookup table's group
    /// (<see cref="TableGroupSuffix"/>) that does not fit the current device goes to the fallback slot and the fill continues where it was;
    /// tables larger than every device do not count toward the shares. <paramref name="used"/> receives the bytes placed on each slot.
    /// </summary>
    private static Dictionary<string, int> AssignGroups(List<string> groupOrder, Dictionary<string, long> groupBytes, PlacementOptions options, int slotCount, int fallbackSlot, out long[] used)
    {
        int nPipe = options.Devices.Count;
        var capacity = new long[nPipe];
        var share = new double[nPipe];
        for (int i = 0; i < nPipe; i++)
        {
            var d = options.Devices[i];
            capacity[i] = Math.Max(0, (d.MemoryLimit ?? d.Device.AvailableMemory) - options.ReservePerDevice);
            share[i] = d.Weight ?? capacity[i];
        }

        long largest = nPipe > 0 ? capacity.Max() : 0;
        bool Offloaded(string g) => fallbackSlot >= 0 && IsTableGroup(g) && groupBytes[g] > largest;
        long total = groupBytes.Where(kv => !Offloaded(kv.Key)).Sum(kv => kv.Value);

        double shareSum = share.Sum();
        var target = new long[nPipe];
        for (int i = 0; i < nPipe; i++)
        {
            target[i] = shareSum > 0 ? (long)Math.Ceiling(total * (share[i] / shareSum)) : 0;
        }

        var groupDevice = new Dictionary<string, int>();
        used = new long[slotCount];
        int cur = 0;
        foreach (var g in groupOrder)
        {
            long b = groupBytes[g];
            if (fallbackSlot >= 0 && IsTableGroup(g) && (cur >= nPipe || used[cur] + b > capacity[cur]))
            {
                groupDevice[g] = fallbackSlot;
                used[fallbackSlot] += b;
                continue;
            }

            while (cur < nPipe && (used[cur] + b > capacity[cur] || (used[cur] > 0 && used[cur] + b / 2 > target[cur] && cur < nPipe - 1)))
            {
                cur++;
            }

            int slot;
            if (cur < nPipe)
            {
                slot = cur;
            }
            else if (fallbackSlot >= 0)
            {
                slot = fallbackSlot;
            }
            else
            {
                throw new InsufficientMemoryException(
                    $"Model does not fit: needs {total / (1024.0 * 1024 * 1024):F2} GiB, group '{g}' ({b / (1024.0 * 1024 * 1024):F2} GiB) has no room and no fallback device is configured.");
            }

            groupDevice[g] = slot;
            used[slot] += b;
        }

        return groupDevice;
    }

    private static bool IsTableGroup(string group) => group.EndsWith(TableGroupSuffix, StringComparison.Ordinal);

    /// <summary>Nodes reading or writing host-only values (strings).</summary>
    private static bool IsHostOnly(Node n) => n.Inputs.Concat(n.Outputs).Any(v => v.Type.Kind == DataTypeKind.String);

    /// <summary>Device of every node: its group's device, the fallback for ops that device cannot run, and views follow their input.</summary>
    private static int[] AssignNodes(ComputeGraph graph, List<DeviceSlot> slots, Dictionary<string, int> groupDevice, int fallbackSlot, Func<Node, string> groupOf)
    {
        var nodeSlot = new int[graph.Nodes.Count];
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            int s = groupDevice[groupOf(n)];
            if (n.Op.Category is not (OpCategory.View or OpCategory.ControlFlow) && !slots[s].Device.Supports(n).IsSupported)
            {
                if (fallbackSlot >= 0 && slots[fallbackSlot].Device.Supports(n).IsSupported)
                {
                    s = fallbackSlot;
                }
                else
                {
                    throw new NotSupportedException($"Op {n.Op.Name} (node {n.Id}) is not supported on {slots[s].Device.Info.Name}: {slots[s].Device.Supports(n).Reason}");
                }
            }

            if (IsHostOnly(n) && slots[s].Device.Info.Kind != DeviceKind.Cpu)
            {
                s = fallbackSlot >= 0 && slots[fallbackSlot].Device.Info.Kind == DeviceKind.Cpu && slots[fallbackSlot].Device.Supports(n).IsSupported
                    ? fallbackSlot
                    : throw new NotSupportedException($"Op {n.Op.Name} (node {n.Id}) handles strings, which only a CPU device can hold, and no CPU fallback runs it.");
            }

            nodeSlot[i] = s;
        }

        if (fallbackSlot >= 0)
        {
            KeepStatesOnOneDevice(graph, slots, nodeSlot, fallbackSlot);
        }

        // View ops follow their input's device to avoid needless transfers.
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            if (n.Op.Category == OpCategory.View && n.Inputs[0].Producer is { } p && !IsHostOnly(n))
            {
                nodeSlot[i] = nodeSlot[IndexOf(graph, p)];
            }
        }

        return nodeSlot;
    }

    /// <summary>
    /// A persistent state (KV cache, recurrent state) lives in exactly one compiled segment, so every node that reads or writes it must run on
    /// the same device: when op support put some of them on the fallback device (attention with ALiBi on a GPU whose kernels lack it, say), the others
    /// that touch the same state follow, as long as the fallback device runs them. A move can split another state, hence the repetition.
    /// </summary>
    private static void KeepStatesOnOneDevice(ComputeGraph graph, List<DeviceSlot> slots, int[] nodeSlot, int fallbackSlot)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var statesOfNode = new Dictionary<StateDefinition, List<int>>();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op.Category is OpCategory.View or OpCategory.ControlFlow)
                {
                    continue;   // views follow their input's device
                }

                foreach (var input in graph.Nodes[i].Inputs)
                {
                    if (input.State is { } state)
                    {
                        if (!statesOfNode.TryGetValue(state, out var users))
                        {
                            statesOfNode[state] = users = [];
                        }

                        users.Add(i);
                    }
                }
            }

            foreach (var users in statesOfNode.Values)
            {
                if (users.Select(i => nodeSlot[i]).Distinct().Count() < 2)
                {
                    continue;
                }

                foreach (int i in users.Where(i => nodeSlot[i] != fallbackSlot))
                {
                    var node = graph.Nodes[i];
                    if (!slots[fallbackSlot].Device.Supports(node).IsSupported)
                    {
                        throw new NotSupportedException($"Op {node.Op.Name} (node {node.Id}) shares a state with nodes that only run on {slots[fallbackSlot].Device.Info.Name}, which does not support it: {slots[fallbackSlot].Device.Supports(node).Reason}");
                    }

                    nodeSlot[i] = fallbackSlot;
                    changed = true;
                }
            }
        }
    }

    private static int IndexOf(ComputeGraph g, Node n)
    {
        // Node ids are assigned sequentially by the builder, but be robust.
        if (n.Id < g.Nodes.Count && g.Nodes[n.Id] == n)
        {
            return n.Id;
        }

        for (int i = 0; i < g.Nodes.Count; i++)
        {
            if (g.Nodes[i] == n)
            {
                return i;
            }
        }

        throw new InvalidOperationException("Node not in graph.");
    }

    /// <summary>
    /// Device memory a graph needs when compiled with <paramref name="options"/>, before any device is touched: weights, persistent
    /// state (KV caches / recurrent state) and a rough bound for activations and scratch. <paramref name="slots"/> &gt; 1 sizes
    /// the state of a multi-sequence compilation of the single-sequence <paramref name="graph"/> (KV rows for slots · context,
    /// recurrent state per slot). Use it to decide whether a model fits before compiling it.
    /// <paramref name="kvCacheRows"/> &gt; 0 sizes the KV caches of a paged multi-sequence compilation instead (rows shared by all slots).
    /// </summary>
    public static ModelFootprint Estimate(ComputeGraph graph, CompileOptions options, int slots = 1, long kvCacheRows = 0)
    {
        long weights = 0;
        long states = 0;
        long maxOut = 0;
        long maxNode = 0;
        var seenW = new HashSet<WeightInfo>();
        var seenS = new HashSet<StateDefinition>();
        foreach (var n in graph.Nodes)
        {
            long nodeBytes = 0;
            foreach (var v in n.Inputs)
            {
                if (v.Weight is { } w && seenW.Add(w))
                {
                    weights += w.ByteSize;
                }

                if (v.State is { } st && seenS.Add(st))
                {
                    if (slots > 1 && MultiSequence.IsKvState(st))
                    {
                        long rows = kvCacheRows > 0 ? kvCacheRows : options.ContextLength * slots;
                        states += StateBytes(st, new CompileOptions { MaxTokens = options.MaxTokens, ContextLength = rows, SymbolBounds = options.SymbolBounds });
                    }
                    else
                    {
                        states += StateBytes(st, options) * Math.Max(1, slots);
                    }
                }
            }
            if (n.Op.Category == OpCategory.View)
            {
                continue;
            }

            foreach (var o in n.Outputs)
            {
                long b = ValueBytes(o, options);
                nodeBytes += b;
                maxOut = Math.Max(maxOut, b);
            }
            maxNode = Math.Max(maxNode, nodeBytes);
        }
        // Activations live in a liveness-packed arena plus staging copies of the inputs/outputs; a few times the largest node
        // output (matmul in/out, residual, norm) covers the arena. Device memory is sub-allocated from 256 MiB blocks (the last one
        // is partly empty) and big models need a little extra scratch, hence the granule and a few percent of the resident size.
        long act = 4 * maxOut + (256L << 20) + (weights + states) / 25;
        return new ModelFootprint(weights, states, act);
    }

    private static long ValueBytes(Value v, CompileOptions o)
    {
        long elems = 1;
        foreach (var d in v.Shape.Dims)
        {
            elems *= d.IsSymbolic ? Math.Max(1, o.HasBound(d.Symbol!) ? o.Bound(d.Symbol!) : o.MaxTokens) : d.Fixed;
        }

        return v.Type.ByteSize(elems);
    }

    /// <summary>Bytes of a state with its symbolic dimensions bound by <paramref name="o"/>.</summary>
    public static long StateBytes(StateDefinition s, CompileOptions o)
    {
        long elems = 1;
        foreach (var d in s.Shape.Dims)
        {
            elems *= d.IsSymbolic ? o.Bound(d.Symbol!) : d.Fixed;
        }

        return s.Type.ByteSize(elems);
    }

    /// <summary>
    /// Whether <paramref name="n"/> publishes a data-dependent count (its <c>count_symbol</c> attribute names one of the graph's
    /// dynamic symbols).
    /// </summary>
    private static bool PublishesCount(Node n, ComputeGraph graph)
        => n.Op.Category != OpCategory.ControlFlow && n.Attributes.Get("count_symbol", "") is { Length: > 0 } symbol
           && graph.DynamicSymbols.Any(d => d.Key == symbol);

    private static List<GraphSegment> BuildSegments(ComputeGraph graph, int[] nodeSlot, List<DeviceSlot> slots, bool groupCounts)
    {
        // A GPU learns a data-dependent count only when its segment has finished (the count is read back with the outputs); the CPU
        // publishes it as soon as the node has run. Ordered by count level, the counts of one level share a segment, which ends where
        // the level changes; in graph order (SHARPLIGHT_COUNT_LEVELS=0) a node that publishes a count ends its segment.
        var schedule = groupCounts && !CountLevelSchedule.DisabledByEnvironment
            ? CountLevelSchedule.Build(graph, i => slots[nodeSlot[i]].Device.Info.Kind != DeviceKind.Cpu && graph.Nodes[i].Op.Category != OpCategory.ControlFlow)
            : null;
        var runs = new List<(int Slot, List<Node> Nodes)>();
        int previous = -1;
        foreach (int i in schedule?.Order ?? Enumerable.Range(0, graph.Nodes.Count))
        {
            // A control-flow node runs alone in a host segment.
            bool controlFlow = graph.Nodes[i].Op.Category == OpCategory.ControlFlow;
            bool afterCount = schedule is { } s
                ? previous >= 0 && s.Level[previous] != s.Level[i]
                : runs.Count > 0 && slots[runs[^1].Slot].Device.Info.Kind != DeviceKind.Cpu && PublishesCount(runs[^1].Nodes[^1], graph);
            if (runs.Count == 0 || runs[^1].Slot != nodeSlot[i] || controlFlow || runs[^1].Nodes[^1].Op.Category == OpCategory.ControlFlow || afterCount)
            {
                runs.Add((nodeSlot[i], []));
            }

            runs[^1].Nodes.Add(graph.Nodes[i]);
            previous = i;
        }

        var producerRun = new Dictionary<Value, int>();
        for (int r = 0; r < runs.Count; r++)
        {
            foreach (var n in runs[r].Nodes)
            {
                foreach (var o in n.Outputs)
                {
                    producerRun[o] = r;
                }
            }
        }

        var graphOutputs = graph.Outputs.Values.ToHashSet();
        var segs = new List<GraphSegment>();
        for (int r = 0; r < runs.Count; r++)
        {
            var inputs = new List<Value>();
            var seenIn = new HashSet<Value>();
            foreach (var n in runs[r].Nodes)
            {
                foreach (var v in n.Inputs)
                {
                    if ((v.Kind == ValueKind.Input || (v.Kind == ValueKind.Intermediate && producerRun[v] != r)) && seenIn.Add(v))
                    {
                        inputs.Add(v);
                    }
                }
            }

            var outputs = new List<Value>();
            foreach (var n in runs[r].Nodes)
            {
                foreach (var o in n.Outputs)
                {
                    bool usedLater = graphOutputs.Contains(o);
                    for (int r2 = r + 1; r2 < runs.Count && !usedLater; r2++)
                    {
                        usedLater = runs[r2].Nodes.Any(n2 => n2.Inputs.Contains(o));
                    }

                    if (usedLater)
                    {
                        outputs.Add(o);
                    }
                }
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
