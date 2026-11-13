using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Runtime;

/// <summary>
/// Pipeline-style partitioner. Placement groups (normally one per layer) are assigned to devices in
/// graph order, filling each device up to its share of the total memory. Groups that fit nowhere
/// go to the fallback device (CPU offload); so do individual nodes the chosen device can't run.
/// </summary>
public static class PlacementPlanner
{
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

        // ---- cost per group -------------------------------------------------------------------
        var groupOrder = new List<string>();
        var groupBytes = new Dictionary<string, long>();
        var seenWeights = new HashSet<WeightInfo>();
        var seenStates = new HashSet<StateDefinition>();
        foreach (var n in graph.Nodes)
        {
            if (!groupBytes.ContainsKey(n.Group))
            {
                groupBytes[n.Group] = 0;
                groupOrder.Add(n.Group);
            }
            foreach (var v in n.Inputs)
            {
                if (v.Weight is { } w && seenWeights.Add(w))
                {
                    groupBytes[n.Group] += w.ByteSize;
                }

                if (v.State is { } s && seenStates.Add(s))
                {
                    groupBytes[n.Group] += StateBytes(s, options.Compile);
                }
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
        {
            target[i] = shareSum > 0 ? (long)Math.Ceiling(total * (share[i] / shareSum)) : 0;
        }

        // ---- greedy fill in order -------------------------------------------------------------
        var groupDevice = new Dictionary<string, int>();
        var used = new long[slots.Count];
        int cur = 0;
        foreach (var g in groupOrder)
        {
            long b = groupBytes[g];
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

        // ---- per-node device (op support fallback) -------------------------------------------
        var nodeSlot = new int[graph.Nodes.Count];
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            int s = groupDevice[n.Group];
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
            if (n.Op.Category == OpCategory.View && n.Inputs[0].Producer is { } p)
            {
                nodeSlot[i] = nodeSlot[IndexOf(graph, p)];
            }
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
    /// </summary>
    public static ModelFootprint Estimate(ComputeGraph graph, CompileOptions options, int slots = 1)
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
                        states += StateBytes(st, new CompileOptions { MaxTokens = options.MaxTokens, ContextLength = options.ContextLength * slots, SymbolBounds = options.SymbolBounds });
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

    private static List<GraphSegment> BuildSegments(ComputeGraph graph, int[] nodeSlot)
    {
        var runs = new List<(int Slot, List<Node> Nodes)>();
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            // A control-flow node runs alone in a host segment.
            bool controlFlow = graph.Nodes[i].Op.Category == OpCategory.ControlFlow;
            if (runs.Count == 0 || runs[^1].Slot != nodeSlot[i] || controlFlow || runs[^1].Nodes[^1].Op.Category == OpCategory.ControlFlow)
            {
                runs.Add((nodeSlot[i], []));
            }

            runs[^1].Nodes.Add(graph.Nodes[i]);
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
