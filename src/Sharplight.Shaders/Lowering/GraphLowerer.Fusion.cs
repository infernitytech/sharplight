using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    // ---- units & fusion ----------------------------------------------------------------------
    private static bool IsFusable(Node n) =>
        s_elementwise.Contains(n.Op.Name) && n.Outputs[0].Type == DataType.F32 &&
        n.Inputs.All(i => !i.Type.IsQuantized && (i.Type == DataType.F32 || i.Type == DataType.F16 || i.Type == DataType.BF16));

    /// <summary>
    /// Groups the segment's nodes into scheduling units: elementwise chains become fused groups, and neighbouring
    /// nodes that can run inside another unit's kernel (norms, matmul pairs, rope, ...) are folded into it.
    /// </summary>
    private List<LoweringUnit> FormUnits()
    {
        var consumers = ConsumersByValue();
        var units = GroupElementwiseChains();
        FoldNormTails(units);
        FoldNormHeads(units, consumers);
        if (!DiagnosticSwitches.NoDualMatMul)
        {
            FoldDualMatMuls(units, consumers);
        }

        AbsorbIntoConsumers(units, consumers);
        FoldScatterChains(units, consumers);
        FindFusedOutputs(units, consumers);
        return units;
    }

    /// <summary>For every value, the nodes reading it (a node appears once per input slot that reads the value).</summary>
    private Dictionary<Value, List<Node>> ConsumersByValue()
    {
        var consumers = new Dictionary<Value, List<Node>>();
        foreach (var node in _seg.Nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (!consumers.TryGetValue(input, out var list))
                {
                    consumers[input] = list = [];
                }

                list.Add(node);
            }
        }

        return consumers;
    }

    /// <summary>True when <paramref name="v"/> is a segment output or is read by a node outside <paramref name="group"/>.</summary>
    private bool EscapesGroup(Value v, Dictionary<Value, List<Node>> consumers, HashSet<Node> group)
        => _segmentOutputs.Contains(v) || (consumers.TryGetValue(v, out var readers) && readers.Any(c => !group.Contains(c)));

    /// <summary>One unit per view or non-fusable node; consecutive fusable elementwise nodes of one shape form a group.</summary>
    private List<LoweringUnit> GroupElementwiseChains()
    {
        var units = new List<LoweringUnit>();
        List<Node>? open = null;
        HashSet<Value>? openValues = null;

        void Close()
        {
            if (open is null)
            {
                return;
            }

            units.Add(new LoweringUnit { Fused = open });
            open = null;
            openValues = null;
        }

        foreach (var n in _seg.Nodes)
        {
            if (n.Op.Category == OpCategory.View)
            {
                if (openValues is not null && openValues.Contains(n.Inputs[0]))
                {
                    Close();
                }

                units.Add(new LoweringUnit { Node = n });
                continue;
            }

            if (IsFusable(n))
            {
                bool joins = open is not null && n.Outputs[0].Shape.Equals(open[0].Outputs[0].Shape) && n.Inputs.Any(openValues!.Contains);
                if (!joins)
                {
                    Close();
                    open = [];
                    openValues = [];
                }

                open!.Add(n);
                openValues!.Add(n.Outputs[0]);
                continue;
            }

            Close();
            units.Add(new LoweringUnit { Node = n });
        }

        Close();
        return units;
    }

    /// <summary>True when a row norm's optional weight is a rank-1 f32 vector (the only form the fused kernel handles).</summary>
    private static bool HasFusableNormWeight(Node norm)
        => norm.Inputs.Length <= 1 || (norm.Inputs[1].Type == DataType.F32 && norm.Inputs[1].Shape.Rank == 1);

    /// <summary>
    /// An elementwise group directly followed by a row norm of one of its outputs becomes one kernel
    /// (e.g. residual Add + RmsNorm): saves a dispatch and a barrier.
    /// </summary>
    private static void FoldNormTails(List<LoweringUnit> units)
    {
        for (int i = 0; i + 1 < units.Count; i++)
        {
            if (units[i].Fused is not { } group || units[i + 1].Node is not { } norm || norm.Op.Name is not ("RmsNorm" or "L2Norm"))
            {
                continue;
            }

            var x = norm.Inputs[0];
            if (!group.Any(f => f.Output == x) || !x.Shape.Equals(group[0].Output.Shape) || x.Shape.Rank < 1 || x.Shape[-1].IsSymbolic)
            {
                continue;
            }

            if (!HasFusableNormWeight(norm))
            {
                continue;
            }

            units[i].NormTail = norm;
            units.RemoveAt(i + 1);
        }
    }

    /// <summary>
    /// A row norm whose output is consumed only by the next elementwise group (views of other values may sit in
    /// between) is computed in that group's kernel.
    /// </summary>
    private void FoldNormHeads(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        for (int i = 0; i < units.Count; i++)
        {
            if (units[i].Node is not { Op.Name: "RmsNorm" or "L2Norm" } norm)
            {
                continue;
            }

            int next = i + 1;
            while (next < units.Count && units[next].IsView && !units[next].Node!.Inputs.Contains(norm.Output))
            {
                next++;
            }

            if (next >= units.Count || units[next].Fused is not { } group || units[next].NormTail is not null)
            {
                continue;
            }

            var y = norm.Output;
            if (_segmentOutputs.Contains(y) || !y.Shape.Equals(group[0].Output.Shape) || y.Shape.Rank < 1 || y.Shape[-1].IsSymbolic)
            {
                continue;
            }

            if (!consumers.TryGetValue(y, out var readers) || readers.Any(c => !group.Contains(c)))
            {
                continue;
            }

            if (!HasFusableNormWeight(norm))
            {
                continue;
            }

            units[next].NormHead = norm;
            units.RemoveAt(i);
            i--;
        }
    }

    /// <summary>Gate/up style matmul pairs (sharing x) feeding one elementwise group become one dual matvec unit.</summary>
    private void FoldDualMatMuls(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        var unitOf = new Dictionary<Node, LoweringUnit>();
        foreach (var unit in units)
        {
            if (unit.Node is not null)
            {
                unitOf[unit.Node] = unit;
            }
        }

        var removed = new HashSet<LoweringUnit>();
        foreach (var unit in units)
        {
            if (unit.Fused is not { } group || unit.NormHead is not null || unit.NormTail is not null)
            {
                continue;
            }

            var inGroup = group.ToHashSet();
            var external = group.SelectMany(n => n.Inputs).Where(v => !inGroup.Contains(v.Producer!)).Distinct().ToList();
            if (external.Count != 2 || external[0].Producer is not { Op.Name: "MatMul" } a || external[1].Producer is not { Op.Name: "MatMul" } b)
            {
                continue;
            }

            if (!AreDualMatMulPair(a, b))
            {
                continue;
            }

            if (!unitOf.TryGetValue(a, out var unitA) || !unitOf.TryGetValue(b, out var unitB))
            {
                continue;
            }

            if (external.Any(v => _segmentOutputs.Contains(v) || consumers[v].Any(c => !inGroup.Contains(c))))
            {
                continue;
            }

            var escaping = group.Select(n => n.Output).Where(v => EscapesGroup(v, consumers, inGroup)).ToList();
            if (escaping.Count != 1 || !escaping[0].Shape.Equals(a.Output.Shape) || !group[0].Output.Shape.Equals(a.Output.Shape))
            {
                continue;
            }

            unit.Dual = (a, b);
            removed.Add(unitA);
            removed.Add(unitB);
        }

        units.RemoveAll(removed.Contains);
    }

    /// <summary>Two weight matmuls of the same x, weight type and shape that one dual kernel can compute.</summary>
    private static bool AreDualMatMulPair(Node a, Node b)
    {
        if (a.Inputs[0] != b.Inputs[0] || a.Inputs[1].Type != b.Inputs[1].Type || !a.Inputs[1].Shape.Equals(b.Inputs[1].Shape))
        {
            return false;
        }

        if (a.Inputs[1].Kind != ValueKind.Weight || b.Inputs[1].Kind != ValueKind.Weight || a.Inputs[1].Shape.Rank != 2)
        {
            return false;
        }

        return a.Inputs[1].Shape[1] is { IsSymbolic: false } k && k.Fixed % 8 == 0 && WeightFormat(a.Inputs[1].Type) is not null;
    }

    /// <summary>
    /// Lets a consumer compute the single-use producer in its own kernel: L2-normalized q/k inside a delta rule, an
    /// RmsNorm inside the rope of its row, and a rope inside the cache write that stores it.
    /// </summary>
    private void AbsorbIntoConsumers(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        bool SingleUse(Value v) => consumers.TryGetValue(v, out var readers) && readers.Count == 1 && !_segmentOutputs.Contains(v);

        var absorbed = new HashSet<Node>();
        foreach (var unit in units)
        {
            if (unit.Node is not { Op.Name: "GatedDeltaRule" } deltaRule)
            {
                continue;
            }

            for (int slot = 0; slot < 2; slot++)
            {
                var v = deltaRule.Inputs[slot];
                if (v.Producer is not { Op.Name: "L2Norm" } l2 || !SingleUse(v) || l2.Inputs.Length != 1)
                {
                    continue;
                }

                (unit.Absorbed ??= []).Add(l2);
                absorbed.Add(l2);
                _absorbedNorms[v] = l2;
            }
        }

        // Rope of an RmsNorm-ed row computes the norm itself; a rope whose only consumer is a cache write stores into the cache.
        var unitOfNode = units.Where(u => u.Node is not null).ToDictionary(u => u.Node!);
        foreach (var unit in units)
        {
            if (unit.Node is { Op.Name: "Rope" } rope && rope.Inputs[0] is var rv && rv.Producer is { Op.Name: "RmsNorm" } rmsNorm
                && SingleUse(rv) && rmsNorm.Inputs.Length == 2
                && rmsNorm.Inputs[1].Type == DataType.F32 && rmsNorm.Inputs[1].Shape.Rank == 1 && rv.Shape.Rank == 3 && unitOfNode.ContainsKey(rmsNorm))
            {
                (unit.Absorbed ??= []).Add(rmsNorm);
                absorbed.Add(rmsNorm);
                _absorbedNorms[rv] = rmsNorm;
            }
        }

        foreach (var unit in units)
        {
            if (unit.Node is { Op.Name: "CacheWrite" } cacheWrite && cacheWrite.Inputs[1] is var cv && cv.Producer is { Op.Name: "Rope" } rope
                && SingleUse(cv) && unitOfNode.TryGetValue(rope, out var ropeUnit)
                && cacheWrite.Inputs[0].Shape.Rank == 3 && cv.Shape.Rank == 3
                && cacheWrite.Inputs[0].Shape[1].Equals(cv.Shape[1]) && cacheWrite.Inputs[0].Shape[2].Equals(cv.Shape[2]))
            {
                unit.Absorbed = [.. ropeUnit.Absorbed ?? [], rope, .. unit.Absorbed ?? []];
                absorbed.Add(rope);
                _absorbedRope[cv] = rope;
            }
        }

        units.RemoveAll(u => u.Node is not null && absorbed.Contains(u.Node));
    }

    /// <summary>
    /// Determines the materialized outputs of each fused group. Views were emitted as separate units in between; a
    /// fused group is emitted before any later unit consumes its outputs, which holds because units keep graph order.
    /// </summary>
    private void FindFusedOutputs(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        foreach (var unit in units.Where(u => u.Fused is not null))
        {
            var inGroup = unit.Nodes.ToHashSet();
            unit.FusedOutputs = [.. unit.Fused!.Select(n => n.Outputs[0]).Where(v => EscapesGroup(v, consumers, inGroup))];
        }
    }

    /// <summary>
    /// Orders units by dependency level (stable within a level) so independent launches are recorded next to
    /// each other and share barriers. Units touching the same state keep their relative order.
    /// </summary>
    private static List<LoweringUnit> Schedule(List<LoweringUnit> units)
    {
        var producer = new Dictionary<Value, int>();
        for (int i = 0; i < units.Count; i++)
        {
            foreach (var n in units[i].Nodes)
            {
                foreach (var o in n.Outputs)
                {
                    producer[o] = i;
                }
            }
        }

        var lastState = new Dictionary<object, int>();
        var finish = new int[units.Count];
        for (int i = 0; i < units.Count; i++)
        {
            int lv = 0;
            var states = new List<object>();
            foreach (var n in units[i].Nodes)
            {
                foreach (var v in n.Inputs.Concat(n.Writes))
                {
                    if (producer.TryGetValue(v, out int p) && p != i)
                    {
                        lv = Math.Max(lv, finish[p]);
                    }

                    if (v.Kind == ValueKind.State)
                    {
                        if (lastState.TryGetValue(v.State!, out int q) && q != i)
                        {
                            lv = Math.Max(lv, finish[q]);
                        }

                        states.Add(v.State!);
                    }
                }
            }

            foreach (var st in states)
            {
                lastState[st] = i;
            }

            units[i].Level = lv;
            finish[i] = lv + (units[i].IsView ? 0 : 1);
        }
        if (DiagnosticSwitches.NoSchedule)
        {
            return units;
        }

        return [.. units.Select((u, i) => (u, i)).OrderBy(x => x.u.Level).ThenBy(x => x.i).Select(x => x.u)];
    }

    private void ComputeLiveness(List<LoweringUnit> units)
    {
        // Root of each value: views map to the root of their input; everything else is its own root.
        foreach (var u in units)
        {
            foreach (var n in u.Nodes)
            {
                foreach (var o in n.Outputs)
                {
                    _root[o] = n.Op.Category == OpCategory.View ? RootOf(n.Inputs[0]) : o;
                }
            }
        }

        for (int u = 0; u < units.Count; u++)
        {
            foreach (var n in units[u].Nodes)
            {
                foreach (var o in n.Outputs)
                {
                    _lastUse.TryAdd(RootOf(o), u);
                }

                foreach (var i in n.Inputs)
                {
                    _lastUse[RootOf(i)] = u;
                }
            }
        }

        foreach (var o in _seg.Outputs)
        {
            _lastUse[RootOf(o)] = int.MaxValue;
        }
    }
}
