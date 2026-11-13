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
/// <item>Paged (<see cref="Transform(ComputeGraph, int, int, int)"/> with a block size): the KV caches are pools of blocks of
/// <c>block size</c> rows shared by all slots (compile with context length = pool rows), and the <see cref="InputRole.KvBlockTable"/>
/// input [S, blocks per slot] maps each slot's positions to blocks; CacheWrite and CachedAttention get it as their last input
/// and attribute <c>block_size</c>.</item>
/// <item>Other states (recurrent, conv) get a leading [S] dimension; CausalConv1d / GatedDeltaRule / SelectiveScan / Wkv6 / Wkv7 get the runs table.</item>
/// </list>
/// The <c>n_past</c> scalar is no longer used by the rewritten ops. A slot is cleared by zeroing its recurrent state rows
/// (KV rows need no clearing: attention only reads rows up to each query's position).
/// </summary>
public static class MultiSequence
{
    /// <summary>Name of the <see cref="InputRole.SequenceRows"/> input added by <see cref="Transform"/>.</summary>
    public const string RowsInput = "sequence_rows";
    private const string RunsInput = "sequence_runs";
    /// <summary>Name of the <see cref="InputRole.KvBlockTable"/> input of a paged graph.</summary>
    public const string BlockTableInput = "kv_block_table";

    /// <summary>The rewritten graph and, per original state, its multi-sequence definition.</summary>
    public sealed record Result(ComputeGraph Graph, IReadOnlyDictionary<StateDefinition, StateDefinition> States, int Slots)
    {
        /// <summary>Rows per KV block of a paged graph; 0 when the caches are contiguous per slot.</summary>
        public int BlockSize { get; init; }
        /// <summary>Columns of the block table (blocks per slot) of a paged graph.</summary>
        public int BlocksPerSlot { get; init; }
    }

    private static readonly HashSet<string> s_supported = ["CacheWrite", "CachedAttention", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "Wkv7", "Wkv6"];

    /// <summary>True when every stateful op of the graph has a multi-sequence form.</summary>
    public static bool IsSupported(ComputeGraph g, out string? reason)
    {
        foreach (var n in g.Nodes)
        {
            if ((n.Inputs.Any(v => v.Kind == ValueKind.State) || n.Writes.Length > 0) && !s_supported.Contains(n.Op.Name))
            {
                reason = $"op {n.Op.Name} uses state but has no multi-sequence form";
                return false;
            }
        }

        reason = null;
        return true;
    }

    /// <summary>Whether the state is a KV cache (symbolic outermost dimension) rather than a fixed-size recurrent state.</summary>
    public static bool IsKvState(StateDefinition s) => s.Shape.Rank > 0 && s.Shape[0].IsSymbolic;

    /// <summary>
    /// Rewrites a single-sequence graph so it advances several independent sequences in one pass.
    /// </summary>
    /// <param name="g">The single-sequence graph.</param>
    /// <param name="slots">Number of sequences ("slots") the rewritten graph serves.</param>
    /// <returns>The rewritten graph and the state definitions to allocate.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="slots"/> is below 1.</exception>
    /// <exception cref="NotSupportedException">A stateful op of the graph has no multi-sequence form (see <see cref="IsSupported"/>).</exception>
    public static Result Transform(ComputeGraph g, int slots) => Transform(g, slots, 0, 0);

    /// <summary>
    /// Rewrites a single-sequence graph so it advances several independent sequences in one pass, with paged KV caches when
    /// <paramref name="blockSize"/> is positive.
    /// </summary>
    /// <param name="g">The single-sequence graph.</param>
    /// <param name="slots">Number of sequences ("slots") the rewritten graph serves.</param>
    /// <param name="blockSize">Rows per KV block (0: each slot owns a contiguous range of the caches).</param>
    /// <param name="blocksPerSlot">Block table columns: the most blocks one slot can hold (its context / <paramref name="blockSize"/>).</param>
    /// <returns>The rewritten graph and the state definitions to allocate.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="slots"/> is below 1, or a paged layout without blocks per slot.</exception>
    /// <exception cref="NotSupportedException">A stateful op of the graph has no multi-sequence form (see <see cref="IsSupported"/>).</exception>
    public static Result Transform(ComputeGraph g, int slots, int blockSize, int blocksPerSlot)
    {
        if (slots < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(slots));
        }

        if (blockSize < 0 || (blockSize > 0 && blocksPerSlot < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(blocksPerSlot));
        }

        if (!IsSupported(g, out var why))
        {
            throw new NotSupportedException($"Graph '{g.Name}': {why}.");
        }

        var weights = g.Weights.Where(w => w.Weight is not null).ToDictionary(w => w.Name, w => w.Weight!);
        var b = new GraphBuilder(g.Name, n => weights.GetValueOrDefault(n));
        var map = new Dictionary<Value, Value>();
        var states = new Dictionary<StateDefinition, StateDefinition>();

        foreach (var i in g.Inputs)
        {
            map[i] = b.Input(i.Name, i.Input!.Role, i.Type, i.Shape);
        }

        var T = Dim.Sym(WellKnown.TokenSymbol);
        var rows = b.Input(RowsInput, InputRole.SequenceRows, DataType.I32, new Shape(T, Dim.Of(4)));
        var runs = b.Input(RunsInput, InputRole.SequenceRuns, DataType.I32, new Shape(Dim.Sym(WellKnown.RunSymbol), Dim.Of(4)));
        var pages = blockSize > 0 ? b.Input(BlockTableInput, InputRole.KvBlockTable, DataType.I32, new Shape(Dim.Of(slots), Dim.Of(blocksPerSlot))) : null;
        foreach (var s in g.RuntimeScalars)
        {
            if (s != WellKnown.PastScalar)
            {
                b.RuntimeScalar(s);
            }
        }

        Value Map(Value v)
        {
            if (map.TryGetValue(v, out var m))
            {
                return m;
            }

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
            if (states.TryGetValue(s, out var d))
            {
                return d;
            }

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
                    ins.Add(rows);
                    ins.Add(runs);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "CausalConv1d" or "GatedDeltaRule" or "SelectiveScan" or "Wkv7" or "Wkv6":
                    ins.Add(runs);
                    break;
            }

            if (pages is not null && n.Op.Name is "CacheWrite" or "CachedAttention")
            {
                ins.Add(pages);
                attrs = attrs.With("block_size", (long)blockSize);
            }

            IReadOnlyList<Value> outs;
            using (b.Group(n.Group))
            {
                outs = b.Emit(n.Op, ins, attrs, n.Outputs.Length == 1 ? n.Outputs[0].Name : null, n.Writes.Select(Map).ToList());
            }

            for (int k = 0; k < outs.Count; k++)
            {
                map[n.Outputs[k]] = outs[k];
            }
        }
        foreach (var (name, v) in g.Outputs)
        {
            b.Output(name, Map(v));
        }

        return new Result(b.Build(), states, slots) { BlockSize = blockSize, BlocksPerSlot = blockSize > 0 ? blocksPerSlot : 0 };
    }
}
