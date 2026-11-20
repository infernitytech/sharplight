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
/// With a fallback device, the routed experts of a mixture-of-experts layer (the expert-indexed products whose weight bank holds more
/// experts than a token selects, with the activations between them) also form a group of their own (<see cref="ExpertGroupSuffix"/>).
/// When the model does not fit the devices, the routed experts of the first layers go to the fallback, as few layers as needed, and
/// everything else (attention, shared experts, dense layers, the output head) stays on the devices: a token reads its few selected
/// experts but every row of the dense weights, so the fallback reads the fewest bytes per token
/// (<see cref="PlacementOptions.SplitExperts"/>).
/// </summary>
public static class PlacementPlanner
{
    /// <summary>Suffix of the placement group that holds the gathers of a lookup table, appended to the group of the gathers.</summary>
    public const string TableGroupSuffix = "#table";

    /// <summary>Suffix of the placement group that holds a layer's routed experts, appended to the layer's group.</summary>
    public const string ExpertGroupSuffix = "#experts";

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

        bool splitExperts = fallbackSlot >= 0 && options.SplitExperts && Environment.GetEnvironmentVariable("SHARPLIGHT_SPLIT_EXPERTS") is not ("0" or "false");
        var groupOf = GroupKeys(graph, fallbackSlot >= 0, splitExperts);
        var (groupOrder, groupBytes) = MeasureGroups(graph, options.Compile, groupOf);
        var transfers = splitExperts ? MeasureExpertTransfers(graph, options.Compile, groupOf) : [];
        bool stream = splitExperts && options.StreamExpertsMinRows > 0 && slots[fallbackSlot].Device.Info.Kind == DeviceKind.Cpu
            && options.Devices.Any(d => d.Device.StreamsWeights);
        var groupDevice = AssignGroups(groupOrder, groupBytes, transfers, stream, options, slots.Count, fallbackSlot, out long[] used);
        var nodeSlot = AssignNodes(graph, slots, groupDevice, fallbackSlot, groupOf);
        var segments = BuildSegments(graph, nodeSlot, slots, options.GroupDataDependentCounts);
        return new PlacementPlan
        {
            Slots = slots,
            Segments = segments,
            GroupDevice = groupDevice,
            BytesPerSlot = Enumerable.Range(0, slots.Count).ToDictionary(i => i, i => used[i]),
            StreamedSegments = stream ? StreamedSegments(segments, slots, fallbackSlot, groupOf) : new Dictionary<int, int>(),
        };
    }

    /// <summary>
    /// The placement group of every node: its own group, or with a fallback device, for the gathers of a lookup table (a weight that only
    /// <c>Gather</c> reads, as its table input) that group plus <see cref="TableGroupSuffix"/>, and with <paramref name="splitExperts"/>, for
    /// the routed experts of a layer (<see cref="ExpertNodes"/>) that group plus <see cref="ExpertGroupSuffix"/>.
    /// </summary>
    private static Func<Node, string> GroupKeys(ComputeGraph graph, bool hasFallback, bool splitExperts)
    {
        if (!hasFallback)
        {
            return static n => n.Group;
        }

        var tableKeys = TableGroupKeys(graph);
        if (!splitExperts)
        {
            return tableKeys;
        }

        var experts = ExpertNodes(graph);
        if (experts.Count == 0)
        {
            return tableKeys;
        }

        return n => experts.Contains(n) ? n.Group + ExpertGroupSuffix : tableKeys(n);
    }

    /// <summary>The gathers of lookup tables in groups of their own (<see cref="TableGroupSuffix"/>), every other node in its group.</summary>
    private static Func<Node, string> TableGroupKeys(ComputeGraph graph)
    {
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
        var inTable = new HashSet<Node>(graph.Nodes.Where(n => IsTableGather(n) && n.Inputs[0].Weight is { } table && tables.Contains(table)));
        if (inTable.Count == 0)
        {
            return static n => n.Group;
        }

        // The index computations that only feed a table's gathers (slices, reshapes of the ids) join them, so the table's device does not
        // alternate with the layer's for every gather. Nodes with weights or states, and graph outputs, stay where they are.
        var consumers = new Dictionary<Value, List<Node>>();
        foreach (var n in graph.Nodes)
        {
            foreach (var v in n.Inputs)
            {
                if (!consumers.TryGetValue(v, out var list))
                {
                    consumers[v] = list = [];
                }

                list.Add(n);
            }
        }

        var graphOutputs = graph.Outputs.Values.ToHashSet();
        for (int i = graph.Nodes.Count - 1; i >= 0; i--)
        {
            var n = graph.Nodes[i];
            bool feedsOnlyTables = !inTable.Contains(n) && n.Outputs.Length > 0
                && n.Inputs.All(v => v.Weight is null && v.State is null)
                && n.Outputs.All(v => !graphOutputs.Contains(v) && consumers.TryGetValue(v, out var users) && users.All(u => inTable.Contains(u) && u.Group == n.Group));
            if (feedsOnlyTables)
            {
                inTable.Add(n);
            }
        }

        return n => inTable.Contains(n) ? n.Group + TableGroupSuffix : n.Group;
    }

    private static bool IsTableGather(Node n) => n.Op.Name == "Gather";

    /// <summary>
    /// The routed-expert nodes of every mixture-of-experts layer: the expert-indexed products (<c>MatMulId</c>) whose weight bank holds more
    /// experts than a row selects, and, in the same group, the nodes between them: weightless nodes that read a value they produce, the
    /// expert inputs or weights (activations, slices), and nodes whose outputs only they read (per-expert bias gathers). The router, the
    /// weighting of the slots and the shared experts stay in the layer's group.
    /// </summary>
    private static HashSet<Node> ExpertNodes(ComputeGraph graph)
    {
        var region = new HashSet<Node>(graph.Nodes.Where(IsRoutedExpertProduct));
        if (region.Count == 0)
        {
            return region;
        }

        var groups = region.Select(n => n.Group).ToHashSet();
        var boundary = new HashSet<Value>(region.SelectMany(n => n.Inputs).Where(v => v.Weight is null && (v.Producer is null || !region.Contains(v.Producer))));
        var consumers = new Dictionary<Value, List<Node>>();
        foreach (var n in graph.Nodes)
        {
            foreach (var v in n.Inputs)
            {
                if (!consumers.TryGetValue(v, out var list))
                {
                    consumers[v] = list = [];
                }

                list.Add(n);
            }
        }

        var graphOutputs = graph.Outputs.Values.ToHashSet();
        bool InRegionOrBoundary(Value v) => v.Weight is not null || boundary.Contains(v) || (v.Producer is { } p && region.Contains(p));
        // A value of the layer that only `reader` reads, produced from weights and the region's inputs alone (collected into `side`).
        bool IsSideInput(Value v, Node reader, List<Node> side)
        {
            if (v.Producer is not { } p || region.Contains(p) || !groups.Contains(p.Group) || p.Outputs.Length != 1 || p.Writes.Length > 0
                || p.Op.Category == OpCategory.ControlFlow || graphOutputs.Contains(v) || !consumers.TryGetValue(v, out var users) || users.Any(u => u != reader)
                || !p.Inputs.All(i => i.State is null && (i.Weight is not null || boundary.Contains(i))))
            {
                return false;
            }

            side.Add(p);
            return true;
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            foreach (var n in graph.Nodes)
            {
                if (region.Contains(n) || !groups.Contains(n.Group) || n.Outputs.Length == 0 || n.Writes.Length > 0
                    || n.Op.Category == OpCategory.ControlFlow || n.Inputs.Any(v => v.State is not null) || n.Outputs.Any(graphOutputs.Contains))
                {
                    continue;
                }

                // Forward: a weightless node fed by the region (an activation, a bias add), with the side nodes that only it reads and that
                // read only weights and the region's inputs (the per-expert bias gather before the add). Backward: a node only the region reads.
                var side = new List<Node>();
                bool fedByRegion = n.Inputs.All(v => v.Weight is null) && n.Inputs.Any(v => v.Producer is { } p && region.Contains(p))
                    && n.Inputs.All(v => InRegionOrBoundary(v) || IsSideInput(v, n, side));
                bool onlyFeedsRegion = n.Inputs.All(InRegionOrBoundary) && n.Outputs.All(o => consumers.TryGetValue(o, out var users) && users.All(region.Contains));
                if (fedByRegion || onlyFeedsRegion)
                {
                    region.Add(n);
                    if (fedByRegion)
                    {
                        region.UnionWith(side);
                    }

                    changed = true;
                }
            }
        }

        ExtendToSlotReductions(region, groups, consumers, graphOutputs);
        return region;
    }

    /// <summary>
    /// Adds the weightless chains that reduce the region's per-slot outputs (the weighting of the k slots and their sum), so the fallback
    /// returns one row per token instead of k: a chain is followed for a few nodes of the same group and taken only when it ends in a node with
    /// fewer elements per row than the region value it started from.
    /// </summary>
    private static void ExtendToSlotReductions(HashSet<Node> region, HashSet<string> groups, Dictionary<Value, List<Node>> consumers, HashSet<Value> graphOutputs)
    {
        const int MaxChain = 4;
        var exits = region.SelectMany(n => n.Outputs).Where(o => consumers.TryGetValue(o, out var users) && users.Any(u => !region.Contains(u))).ToList();
        foreach (var start in exits)
        {
            long startElements = RowElements(start.Shape);
            var chain = new List<Node>();
            var value = start;
            while (chain.Count < MaxChain && consumers.TryGetValue(value, out var users) && users.Count == 1)
            {
                var n = users[0];
                if (region.Contains(n) || !groups.Contains(n.Group) || n.Outputs.Length != 1 || n.Writes.Length > 0 || n.Op.Category == OpCategory.ControlFlow
                    || n.Inputs.Any(v => v.Weight is not null || v.State is not null) || graphOutputs.Contains(n.Output))
                {
                    break;
                }

                chain.Add(n);
                if (RowElements(n.Output.Shape) < startElements)
                {
                    region.UnionWith(chain);
                    break;
                }

                value = n.Output;
            }
        }
    }

    /// <summary>Elements per row: the product of the fixed dimensions after the first.</summary>
    private static long RowElements(Shape shape)
    {
        long elements = 1;
        for (int i = 1; i < shape.Rank; i++)
        {
            elements *= shape[i].IsSymbolic ? 1 : shape[i].Fixed;
        }

        return elements;
    }

    /// <summary>
    /// Whether <paramref name="n"/> is an expert-indexed product over a weight bank with more experts than a row selects, so that a row reads
    /// only part of the weight (routed experts, as opposed to a bank every row runs through, such as shared experts with constant ids).
    /// </summary>
    private static bool IsRoutedExpertProduct(Node n)
        => n.Op.Name == "MatMulId" && n.Inputs.Length == 3 && n.Inputs[1].Weight is { } w && w.Shape.Rank == 3 && !w.Shape[0].IsSymbolic
           && n.Inputs[2].Shape is { Rank: 2 } ids && !ids[1].IsSymbolic && w.Shape[0].Fixed > ids[1].Fixed;

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
    /// <summary>
    /// The fallback segments that hold only routed experts (and views of their values), each with the slot of the device before it when that device streams weights and
    /// runs every node of the segment (<see cref="PlacementPlan.StreamedSegments"/>).
    /// </summary>
    private static Dictionary<int, int> StreamedSegments(List<GraphSegment> segments, List<DeviceSlot> slots, int fallbackSlot, Func<Node, string> groupOf)
    {
        var streamed = new Dictionary<int, int>();
        for (int i = 1; i < segments.Count; i++)
        {
            var segment = segments[i];
            int device = segments[i - 1].DeviceSlot;
            if (segment.DeviceSlot != fallbackSlot || device == fallbackSlot || !slots[device].Device.StreamsWeights
                || !segment.Nodes.All(n => (IsExpertGroup(groupOf(n)) || n.Op.Category == OpCategory.View) && slots[device].Device.Supports(n).IsSupported))
            {
                continue;
            }

            streamed[i] = device;
        }

        return streamed;
    }

    private static Dictionary<string, int> AssignGroups(List<string> groupOrder, Dictionary<string, long> groupBytes, Dictionary<string, long> transfers, bool stream, PlacementOptions options, int slotCount, int fallbackSlot, out long[] used)
    {
        var expertGroups = groupOrder.Where(IsExpertGroup).ToList();
        var pinned = new HashSet<string>();
        // Routed experts go to the fallback layer by layer, from the first layer on, until every other group fits the devices. Each moved
        // group adds device buffers for the values it exchanges with its layer, the devices then also hold the graph outputs, and with
        // streaming (prefill on the GPU) the largest moved group's weights; a fill is repeated with the room the groups it moved need.
        // A moved group splits its layer into one more device segment (its own stream, launch recordings and library workspace) and, with
        // streaming, adds a streamed compilation, which holds its own copies of the exchanged values.
        int perGroup = stream ? 2 : 1;
        long Overhead(IReadOnlyCollection<string> moved) => moved.Count == 0 ? 0
            : transfers.GetValueOrDefault(OutputsKey) + moved.Sum(g => transfers.GetValueOrDefault(g) + options.ReservePerSegment) * perGroup
              + (stream ? Math.Clamp(options.StreamBuffers, 1, 2) * moved.Max(g => groupBytes[g]) : 0);
        List<string> Moved(Dictionary<string, int> placed) => [.. placed.Where(kv => kv.Value == fallbackSlot && IsExpertGroup(kv.Key)).Select(kv => kv.Key)];
        Dictionary<string, int> groupDevice;
        while (true)
        {
            groupDevice = FillGroups(groupOrder, groupBytes, options, slotCount, fallbackSlot, pinned, Overhead(pinned), out used);
            for (int repeat = 0; repeat < 3; repeat++)
            {
                var moved = Moved(groupDevice);
                var refill = FillGroups(groupOrder, groupBytes, options, slotCount, fallbackSlot, pinned, Overhead(moved), out var refillUsed);
                bool stable = Moved(refill).ToHashSet().SetEquals(moved);
                groupDevice = refill;
                used = refillUsed;
                if (stable)
                {
                    break;
                }
            }

            if (!SpillsBesidesExperts(groupDevice, fallbackSlot) || pinned.Count == expertGroups.Count)
            {
                break;
            }

            pinned.Add(expertGroups[pinned.Count]);
        }

        return groupDevice;
    }

    /// <summary>Key of <see cref="MeasureExpertTransfers"/>'s entry for the graph outputs.</summary>
    private const string OutputsKey = "";

    /// <summary>
    /// Bytes at the compile bounds of the values each routed-expert group exchanges with the rest of the graph (they need device buffers on
    /// the other side when the group runs on the fallback), and under <see cref="OutputsKey"/> the bytes of the graph outputs.
    /// </summary>
    private static Dictionary<string, long> MeasureExpertTransfers(ComputeGraph graph, CompileOptions compile, Func<Node, string> groupOf)
    {
        var transfers = new Dictionary<string, long> { [OutputsKey] = graph.Outputs.Values.Distinct().Sum(v => ValueBytes(v, compile)) };
        var groupOfValue = new Dictionary<Value, string>();
        foreach (var n in graph.Nodes)
        {
            foreach (var o in n.Outputs)
            {
                groupOfValue[o] = groupOf(n);
            }
        }

        var counted = new HashSet<(string, Value)>();
        foreach (var n in graph.Nodes)
        {
            string group = groupOf(n);
            foreach (var v in n.Inputs)
            {
                if (v.Weight is not null || v.State is not null || !groupOfValue.TryGetValue(v, out string? producer) || producer == group)
                {
                    continue;
                }

                // A value crossing into or out of an expert group.
                foreach (string g in new[] { group, producer }.Where(IsExpertGroup))
                {
                    if (counted.Add((g, v)))
                    {
                        transfers[g] = transfers.GetValueOrDefault(g) + ValueBytes(v, compile);
                    }
                }
            }
        }

        return transfers;
    }

    /// <summary>Whether a group other than routed experts or lookup tables went to the fallback slot.</summary>
    private static bool SpillsBesidesExperts(Dictionary<string, int> groupDevice, int fallbackSlot)
        => fallbackSlot >= 0 && groupDevice.Any(kv => kv.Value == fallbackSlot && !IsExpertGroup(kv.Key) && !IsTableGroup(kv.Key));

    private static bool IsExpertGroup(string group) => group.EndsWith(ExpertGroupSuffix, StringComparison.Ordinal);

    /// <summary>
    /// One greedy fill (see <see cref="AssignGroups"/>) with the <paramref name="pinned"/> groups on the fallback slot; they do not count toward
    /// the shares. <paramref name="overhead"/> bytes are kept free on every device.
    /// </summary>
    private static Dictionary<string, int> FillGroups(List<string> groupOrder, Dictionary<string, long> groupBytes, PlacementOptions options, int slotCount, int fallbackSlot, HashSet<string> pinned, long overhead, out long[] used)
    {
        int nPipe = options.Devices.Count;
        var capacity = new long[nPipe];
        var share = new double[nPipe];
        for (int i = 0; i < nPipe; i++)
        {
            var d = options.Devices[i];
            capacity[i] = Math.Max(0, (d.MemoryLimit ?? d.Device.AvailableMemory) - options.ReservePerDevice - overhead);
            share[i] = d.Weight ?? capacity[i];
        }

        long largest = nPipe > 0 ? capacity.Max() : 0;
        bool Offloaded(string g) => fallbackSlot >= 0 && IsTableGroup(g) && groupBytes[g] > largest;
        long total = groupBytes.Where(kv => !Offloaded(kv.Key) && !pinned.Contains(kv.Key)).Sum(kv => kv.Value);

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
            if (pinned.Contains(g) || (fallbackSlot >= 0 && IsTableGroup(g) && (cur >= nPipe || used[cur] + b > capacity[cur])))
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
    /// <paramref name="kvCacheRows"/> &gt; 0 sizes the KV caches of a paged multi-sequence compilation instead (rows shared by all slots);
    /// <paramref name="windowCacheRows"/> &gt; 0 sizes the caches only sliding-window attention reads (<see cref="MultiSequence.WindowedCaches"/>)
    /// at that many rows.
    /// </summary>
    public static ModelFootprint Estimate(ComputeGraph graph, CompileOptions options, int slots = 1, long kvCacheRows = 0, long windowCacheRows = 0)
    {
        var windowed = windowCacheRows > 0 && kvCacheRows > 0 ? MultiSequence.WindowedCaches(graph).Caches : [];
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
                        long rows = windowed.Contains(st) ? windowCacheRows : kvCacheRows > 0 ? kvCacheRows : options.ContextLength * slots;
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
