using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>
/// Turns a single-sequence decoder graph into one that advances several independent sequences ("slots") in one pass
/// (continuous batching). Architecture independent: it rewrites the stateful ops, not the model.
/// <list type="bullet">
/// <item>Adds inputs <see cref="InputRole.SequenceRows"/> [T, 4] and <see cref="InputRole.SequenceRuns"/> [U, 4] (see their docs).
/// Rows of one slot form a run of consecutive rows in context order; positions inputs (rope) carry each row's own position.</item>
/// <item>KV caches keep their [Ctx, ...] shape: compile with context length S · per-slot context; slot s owns rows
/// [s · Ctx/S, (s + 1) · Ctx/S). CacheWrite gets the rows table, CachedAttention rows and runs tables; both attribute <c>slots</c>.</item>
/// <item>Other states (recurrent, conv) get a leading [S] dimension; CausalConv1d / GatedDeltaRule get the runs table.</item>
/// </list>
/// The <c>n_past</c> scalar is no longer used by the rewritten ops. A slot is cleared by zeroing its recurrent state rows
/// (KV rows need no clearing: attention only reads rows up to each query's position).
/// </summary>
public static class MultiSequence
{
    public const string RowsInput = "sequence_rows";
    public const string RunsInput = "sequence_runs";

    /// <summary>The rewritten graph and, per original state, its multi-sequence definition.</summary>
    public sealed record Result(ComputeGraph Graph, IReadOnlyDictionary<StateDefinition, StateDefinition> States, int Slots);

    static readonly HashSet<string> s_supported = ["CacheWrite", "CachedAttention", "CausalConv1d", "GatedDeltaRule"];

    /// <summary>True when every stateful op of the graph has a multi-sequence form.</summary>
    public static bool IsSupported(ComputeGraph g, out string? reason)
    {
        foreach (var n in g.Nodes)
            if ((n.Inputs.Any(v => v.Kind == ValueKind.State) || n.Writes.Length > 0) && !s_supported.Contains(n.Op.Name))
            {
                reason = $"op {n.Op.Name} uses state but has no multi-sequence form";
                return false;
            }
        reason = null;
        return true;
    }

    public static bool IsKvState(StateDefinition s) => s.Shape.Rank > 0 && s.Shape[0].IsSymbolic;

    public static Result Transform(ComputeGraph g, int slots)
    {
        if (slots < 1) throw new ArgumentOutOfRangeException(nameof(slots));
        if (!IsSupported(g, out var why)) throw new NotSupportedException($"Graph '{g.Name}': {why}.");
        var weights = g.Weights.Where(w => w.Weight is not null).ToDictionary(w => w.Name, w => w.Weight!);
        var b = new GraphBuilder(g.Name, n => weights.GetValueOrDefault(n));
        var map = new Dictionary<Value, Value>();
        var states = new Dictionary<StateDefinition, StateDefinition>();

        foreach (var i in g.Inputs) map[i] = b.Input(i.Name, i.Input!.Role, i.Type, i.Shape);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var rows = b.Input(RowsInput, InputRole.SequenceRows, DataType.I32, new Shape(T, Dim.Of(4)));
        var runs = b.Input(RunsInput, InputRole.SequenceRuns, DataType.I32, new Shape(Dim.Sym(WellKnown.RunSymbol), Dim.Of(4)));
        foreach (var s in g.RuntimeScalars) if (s != WellKnown.PastScalar) b.RuntimeScalar(s);

        Value Map(Value v)
        {
            if (map.TryGetValue(v, out var m)) return m;
            m = v.Kind switch
            {
                ValueKind.Weight => b.Weight(v.Name),
                ValueKind.Constant => b.Constant(v.Name, v.Type, v.Shape, v.ConstantData!),
                ValueKind.State => b.State(StateFor(v.State!)),
                _ => throw new InvalidOperationException($"Value {v} has no mapping (not produced before use)."),
            };
            return map[v] = m;
        }

        StateDefinition StateFor(StateDefinition s)
        {
            if (states.TryGetValue(s, out var d)) return d;
            d = IsKvState(s) ? s : new StateDefinition
            {
                Name = s.Name, Type = s.Type, Kind = s.Kind, Group = s.Group,
                Shape = new Shape([Dim.Of(slots), .. s.Shape.Dims]),
            };
            return states[s] = d;
        }

        foreach (var n in g.Nodes)
        {
            var ins = n.Inputs.Select(Map).ToList();
            var attrs = n.Attributes;
            switch (n.Op.Name)
            {
                case "CacheWrite":
                    ins.Add(rows);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "CachedAttention":
                    ins.Add(rows); ins.Add(runs);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "CausalConv1d" or "GatedDeltaRule":
                    ins.Add(runs);
                    break;
            }
            IReadOnlyList<Value> outs;
            using (b.Group(n.Group))
                outs = b.Emit(n.Op, ins, attrs, n.Outputs.Length == 1 ? n.Outputs[0].Name : null, n.Writes.Select(Map).ToList());
            for (int k = 0; k < outs.Count; k++) map[n.Outputs[k]] = outs[k];
        }
        foreach (var (name, v) in g.Outputs) b.Output(name, Map(v));
        return new Result(b.Build(), states, slots);
    }
}
