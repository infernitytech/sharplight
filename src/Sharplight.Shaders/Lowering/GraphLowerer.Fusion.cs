using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;


public sealed partial class GraphLowerer
{
    // ---- units & fusion ----------------------------------------------------------------------
    private static bool IsFusable(Node n) =>
        s_elementwise.Contains(n.Op.Name) && n.Outputs[0].Type == DataType.F32 &&
        n.Inputs.All(i => !i.Type.IsQuantized && (i.Type == DataType.F32 || i.Type == DataType.F16 || i.Type == DataType.BF16));

    private List<LoweringUnit> FormUnits()
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

        // An elementwise group directly followed by a row norm of one of its outputs becomes one kernel
        // (e.g. residual Add + RmsNorm): saves a dispatch and a barrier.
        for (int i = 0; i + 1 < units.Count; i++)
        {
            if (units[i].Fused is not { } g || units[i + 1].Node is not { } nn || nn.Op.Name is not ("RmsNorm" or "L2Norm"))
            {
                continue;
            }

            var x = nn.Inputs[0];
            if (!g.Any(f => f.Output == x) || !x.Shape.Equals(g[0].Output.Shape) || x.Shape.Rank < 1 || x.Shape[-1].IsSymbolic)
            {
                continue;
            }

            if (nn.Inputs.Length > 1 && (nn.Inputs[1].Type != DataType.F32 || nn.Inputs[1].Shape.Rank != 1))
            {
                continue;
            }

            units[i].NormTail = nn;
            units.RemoveAt(i + 1);
        }

        // A row norm whose output is consumed only by the next elementwise group (views of other values may sit in
        // between) is computed in that group's kernel.
        {
            var uses = new Dictionary<Value, List<Node>>();
            foreach (var n in _seg.Nodes)
            {
                foreach (var i in n.Inputs)
                {
                    (uses.TryGetValue(i, out var l) ? l : uses[i] = []).Add(n);
                }
            }

            for (int i = 0; i < units.Count; i++)
            {
                if (units[i].Node is not { Op.Name: "RmsNorm" or "L2Norm" } nn)
                {
                    continue;
                }

                int j = i + 1;
                while (j < units.Count && units[j].IsView && !units[j].Node!.Inputs.Contains(nn.Output))
                {
                    j++;
                }

                if (j >= units.Count || units[j].Fused is not { } g || units[j].NormTail is not null)
                {
                    continue;
                }

                var y = nn.Output;
                if (_segmentOutputs.Contains(y) || !y.Shape.Equals(g[0].Output.Shape) || y.Shape.Rank < 1 || y.Shape[-1].IsSymbolic)
                {
                    continue;
                }

                if (!uses.TryGetValue(y, out var us) || us.Any(c => !g.Contains(c)))
                {
                    continue;
                }

                if (nn.Inputs.Length > 1 && (nn.Inputs[1].Type != DataType.F32 || nn.Inputs[1].Shape.Rank != 1))
                {
                    continue;
                }

                units[j].NormHead = nn;
                units.RemoveAt(i);
                i--;
            }
        }

        // Gate/up style matmul pairs feeding one elementwise group.
        if (!DiagnosticSwitches.NoDualMatMul)
        {
            var uses = new Dictionary<Value, List<Node>>();
            foreach (var n in _seg.Nodes)
            {
                foreach (var i in n.Inputs)
                {
                    (uses.TryGetValue(i, out var l) ? l : uses[i] = []).Add(n);
                }
            }

            var unitOf = new Dictionary<Node, LoweringUnit>();
            foreach (var u in units)
            {
                if (u.Node is not null)
                {
                    unitOf[u.Node] = u;
                }
            }

            var removed = new HashSet<LoweringUnit>();
            foreach (var u in units)
            {
                if (u.Fused is not { } g || u.NormHead is not null || u.NormTail is not null)
                {
                    continue;
                }

                var inGroup = g.ToHashSet();
                var ext = g.SelectMany(n => n.Inputs).Where(v => !inGroup.Contains(v.Producer!)).Distinct().ToList();
                if (ext.Count != 2 || ext[0].Producer is not { Op.Name: "MatMul" } a || ext[1].Producer is not { Op.Name: "MatMul" } b)
                {
                    continue;
                }

                if (a.Inputs[0] != b.Inputs[0] || a.Inputs[1].Type != b.Inputs[1].Type || !a.Inputs[1].Shape.Equals(b.Inputs[1].Shape))
                {
                    continue;
                }

                if (a.Inputs[1].Kind != ValueKind.Weight || b.Inputs[1].Kind != ValueKind.Weight || a.Inputs[1].Shape.Rank != 2)
                {
                    continue;
                }

                if (a.Inputs[1].Shape[1] is not { IsSymbolic: false } kd || kd.Fixed % 8 != 0 || WeightFormat(a.Inputs[1].Type) is null)
                {
                    continue;
                }

                if (!unitOf.TryGetValue(a, out var ua) || !unitOf.TryGetValue(b, out var ub))
                {
                    continue;
                }

                if (ext.Any(v => _segmentOutputs.Contains(v) || uses[v].Any(c => !inGroup.Contains(c))))
                {
                    continue;
                }

                var outs = g.Select(n => n.Output).Where(v => _segmentOutputs.Contains(v) || (uses.TryGetValue(v, out var cs) && cs.Any(c => !inGroup.Contains(c)))).ToList();
                if (outs.Count != 1 || !outs[0].Shape.Equals(a.Output.Shape) || g[0].Output.Shape.Equals(a.Output.Shape) is false)
                {
                    continue;
                }

                u.Dual = (a, b);
                removed.Add(ua); removed.Add(ub);
            }
            units.RemoveAll(removed.Contains);
        }

        // L2-normalized q/k consumed only by a delta rule are normalized inside the delta-rule kernel.
        var users = new Dictionary<Value, int>();
        foreach (var n in _seg.Nodes)
        {
            foreach (var i in n.Inputs)
            {
                users[i] = users.GetValueOrDefault(i) + 1;
            }
        }

        var absorbed = new HashSet<Node>();
        foreach (var u in units)
        {
            if (u.Node is not { Op.Name: "GatedDeltaRule" } dn)
            {
                continue;
            }

            for (int idx = 0; idx < 2; idx++)
            {
                var v = dn.Inputs[idx];
                if (v.Producer is not { Op.Name: "L2Norm" } ln || users.GetValueOrDefault(v) != 1 || _segmentOutputs.Contains(v) || ln.Inputs.Length != 1)
                {
                    continue;
                }

                (u.Absorbed ??= []).Add(ln);
                absorbed.Add(ln);
                _absorbedNorms[v] = ln;
            }
        }
        // Rope of an RmsNorm-ed row computes the norm itself; a rope whose only consumer is a cache write stores into the cache.
        var unitOfNode = units.Where(u => u.Node is not null).ToDictionary(u => u.Node!);
        foreach (var u in units)
        {
            if (u.Node is { Op.Name: "Rope" } r && r.Inputs[0] is var rv && rv.Producer is { Op.Name: "RmsNorm" } rn
                && users.GetValueOrDefault(rv) == 1 && !_segmentOutputs.Contains(rv) && rn.Inputs.Length == 2
                && rn.Inputs[1].Type == DataType.F32 && rn.Inputs[1].Shape.Rank == 1 && rv.Shape.Rank == 3 && unitOfNode.ContainsKey(rn))
            {
                (u.Absorbed ??= []).Add(rn);
                absorbed.Add(rn);
                _absorbedNorms[rv] = rn;
            }
        }
        foreach (var u in units)
        {
            if (u.Node is { Op.Name: "CacheWrite" } cw && cw.Inputs[1] is var cv && cv.Producer is { Op.Name: "Rope" } r
                && users.GetValueOrDefault(cv) == 1 && !_segmentOutputs.Contains(cv) && unitOfNode.TryGetValue(r, out var ru)
                && cw.Inputs[0].Shape.Rank == 3 && cv.Shape.Rank == 3 && cw.Inputs[0].Shape[1].Equals(cv.Shape[1]) && cw.Inputs[0].Shape[2].Equals(cv.Shape[2]))
            {
                u.Absorbed = [.. ru.Absorbed ?? [], r, .. u.Absorbed ?? []];
                absorbed.Add(r);
                _absorbedRope[cv] = r;
            }
        }
        units.RemoveAll(u => u.Node is not null && absorbed.Contains(u.Node));

        // Views were emitted as separate units in between; a fused group must be emitted before any
        // later unit consumes its outputs, which holds because units keep graph order.
        // Determine materialized outputs of each fused group.
        var consumers = new Dictionary<Value, List<Node>>();
        foreach (var n in _seg.Nodes)
        {
            foreach (var i in n.Inputs)
            {
                (consumers.TryGetValue(i, out var l) ? l : consumers[i] = []).Add(n);
            }
        }

        foreach (var u in units.Where(u => u.Fused is not null))
        {
            var inGroup = u.Nodes.ToHashSet();
            u.FusedOutputs = [.. u.Fused!.Select(n => n.Outputs[0]).Where(v =>
                _segmentOutputs.Contains(v) || (consumers.TryGetValue(v, out var cs) && cs.Any(c => !inGroup.Contains(c))))];
        }
        return units;
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
