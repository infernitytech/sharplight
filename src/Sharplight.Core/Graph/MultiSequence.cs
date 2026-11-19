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
/// [s · Ctx/S, (s + 1) · Ctx/S). CacheWrite gets the rows table, CachedAttention and IndexerTopK rows and runs tables; all attribute <c>slots</c>.
/// A bidirectional CachedAttention (<c>causal</c> false) gets the <see cref="InputRole.SequenceRunEndRows"/> table in place of the rows table, so
/// each row sees its slot's keys up to the last row of its run.</item>
/// <item>Paged (<see cref="Transform(ComputeGraph, int, int, int)"/> with a block size): the KV caches are pools of blocks of
/// <c>block size</c> rows shared by all slots (compile with context length = pool rows), and the <see cref="InputRole.KvBlockTable"/>
/// input [S, blocks per slot] maps each slot's positions to blocks; CacheWrite and CachedAttention get it as their last input
/// and attribute <c>block_size</c>.</item>
/// <item>Compressed KV caches (<see cref="StateDefinition.Compression"/>) are resized to equal slot ranges of whole entries, or to the
/// paged pool's ceil(block size / compression) rows per block.</item>
/// <item>Other states (recurrent, conv) get a leading [S] dimension; CausalConv1d / GatedDeltaRule / SelectiveScan / Wkv6 / Wkv7 get the runs table,
/// WindowPool the rows and runs tables.</item>
/// <item>With rollback rows (<see cref="Transform(ComputeGraph, int, int, int, int)"/>): the recurrent ops take their rollback form
/// (<see cref="RecurrentRollback"/>): the <see cref="InputRole.SequenceCommits"/> input and one record state per op, so a slot's
/// tentative (speculative) rows can be rolled back.</item>
/// <item>With spare state rows (<see cref="Transform(ComputeGraph, int, int, int, int, int)"/>): the recurrent ops take their snapshot form
/// when every op with per-slot state has one (<see cref="RecurrentSnapshots"/>): the <see cref="InputRole.SequenceSnapshots"/> input, so a
/// step can keep the state at a row inside a run in a spare row.</item>
/// </list>
/// The <c>n_past</c> scalar is no longer used by the rewritten ops. A slot is cleared by zeroing its recurrent state rows
/// (KV rows need no clearing: attention only reads rows up to each query's position).
/// </summary>
public static class MultiSequence
{
    /// <summary>Name of the <see cref="InputRole.SequenceRows"/> input added by <see cref="Transform(ComputeGraph, int, int, int)"/>.</summary>
    public const string RowsInput = "sequence_rows";
    private const string RunsInput = "sequence_runs";
    /// <summary>Name of the <see cref="InputRole.SequenceRunEndRows"/> input of a graph with bidirectional cached attention.</summary>
    public const string RunEndRowsInput = "sequence_run_end_rows";
    /// <summary>Name of the <see cref="InputRole.KvBlockTable"/> input of a paged graph.</summary>
    public const string BlockTableInput = "kv_block_table";

    /// <summary>The rewritten graph and, per original state, its multi-sequence definition.</summary>
    public sealed record Result(ComputeGraph Graph, IReadOnlyDictionary<StateDefinition, StateDefinition> States, int Slots)
    {
        /// <summary>Rows per KV block of a paged graph; 0 when the caches are contiguous per slot.</summary>
        public int BlockSize { get; init; }
        /// <summary>Columns of the block table (blocks per slot) of a paged graph.</summary>
        public int BlocksPerSlot { get; init; }
        /// <summary>Rows of one tentative run the recurrent ops can roll back (0: no rollback form, or no recurrent op).</summary>
        public int RollbackRows { get; init; }
        /// <summary>
        /// Whether the recurrent ops are in snapshot form (<see cref="RecurrentSnapshots"/>): a step stores the state at a row inside a run
        /// into a spare state row (the graph has the <see cref="InputRole.SequenceSnapshots"/> input).
        /// </summary>
        public bool StepSnapshots { get; init; }
    }

    private static readonly HashSet<string> s_supported = ["CacheWrite", "CachedAttention", "IndexerTopK", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "Wkv7", "Wkv6", "WindowPool"];

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
    public static Result Transform(ComputeGraph g, int slots, int blockSize, int blocksPerSlot) => Transform(g, slots, blockSize, blocksPerSlot, 0);

    /// <summary>
    /// Rewrites a single-sequence graph so it advances several independent sequences in one pass, with paged KV caches when
    /// <paramref name="blockSize"/> is positive and recurrent ops in their rollback form when <paramref name="rollbackRows"/> is.
    /// </summary>
    /// <param name="g">The single-sequence graph.</param>
    /// <param name="slots">Number of sequences ("slots") the rewritten graph serves.</param>
    /// <param name="blockSize">Rows per KV block (0: each slot owns a contiguous range of the caches).</param>
    /// <param name="blocksPerSlot">Block table columns: the most blocks one slot can hold (its context / <paramref name="blockSize"/>).</param>
    /// <param name="rollbackRows">
    /// Most rows of one tentative run whose recurrent state can be rolled back (speculative verification: pending token plus
    /// drafts); 0 for none. Ignored when the graph has no recurrent op.
    /// </param>
    /// <returns>The rewritten graph and the state definitions to allocate.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="slots"/> is below 1, a paged layout without blocks per slot, or negative rollback rows.</exception>
    /// <exception cref="NotSupportedException">A stateful op of the graph has no multi-sequence form (see <see cref="IsSupported"/>).</exception>
    public static Result Transform(ComputeGraph g, int slots, int blockSize, int blocksPerSlot, int rollbackRows) => Transform(g, slots, blockSize, blocksPerSlot, rollbackRows, 0);

    /// <summary>
    /// Like <see cref="Transform(ComputeGraph, int, int, int, int)"/>, with <paramref name="spareStateRows"/> extra rows after the
    /// <paramref name="slots"/> slot rows of every recurrent state (<c>[slots + spare, ...]</c>). No run addresses them; a runtime copies
    /// slot rows into them and back (prefix-cache snapshots of recurrent state, see <see cref="Runtime.CompiledModel.CopyStateRows"/>).
    /// When every op with per-slot state is a recurrent op with a snapshot form, the ops take it (<see cref="RecurrentSnapshots"/>,
    /// <see cref="Result.StepSnapshots"/>), so a step can also store the state at a row inside a run into a spare row.
    /// </summary>
    /// <param name="g">The single-sequence graph.</param>
    /// <param name="slots">Number of sequences ("slots") the rewritten graph serves.</param>
    /// <param name="blockSize">Rows per KV block (0: each slot owns a contiguous range of the caches).</param>
    /// <param name="blocksPerSlot">Block table columns: the most blocks one slot can hold.</param>
    /// <param name="rollbackRows">Most rows of one tentative run whose recurrent state can be rolled back; 0 for none.</param>
    /// <param name="spareStateRows">Rows added to each recurrent state after the slots' (0 for none).</param>
    /// <returns>The rewritten graph and the state definitions to allocate.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="slots"/> is below 1, a paged layout without blocks per slot, or negative rollback or spare rows.</exception>
    /// <exception cref="NotSupportedException">A stateful op of the graph has no multi-sequence form (see <see cref="IsSupported"/>).</exception>
    public static Result Transform(ComputeGraph g, int slots, int blockSize, int blocksPerSlot, int rollbackRows, int spareStateRows)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(spareStateRows);
        if (slots < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(slots));
        }

        if (blockSize < 0 || (blockSize > 0 && blocksPerSlot < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(blocksPerSlot));
        }

        if (rollbackRows < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rollbackRows));
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
        var runEndRows = g.Nodes.Any(n => n.Op.Name == "CachedAttention" && !n.Attributes.Get("causal", true))
            ? b.Input(RunEndRowsInput, InputRole.SequenceRunEndRows, DataType.I32, new Shape(T, Dim.Of(4)))
            : null;
        bool rollback = rollbackRows > 0 && g.Nodes.Any(n => RecurrentRollback.Ops.Contains(n.Op.Name));
        var commits = rollback
            ? b.Input(RecurrentRollback.CommitsInput, InputRole.SequenceCommits, DataType.I32, new Shape(Dim.Sym(WellKnown.RunSymbol), Dim.Of(RecurrentRollback.CommitColumns)))
            : null;
        bool stepSnapshots = spareStateRows > 0 && HasStepSnapshotForm(g);
        var snapshots = stepSnapshots
            ? b.Input(RecurrentSnapshots.Input, InputRole.SequenceSnapshots, DataType.I32, new Shape(Dim.Sym(WellKnown.RunSymbol), Dim.Of(RecurrentSnapshots.Columns)))
            : null;
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

            d = IsKvState(s) ? s.Compression > 1 ? CompressedState(s, slots, blockSize) : s : new StateDefinition
            {
                Name = s.Name, Type = s.Type, Kind = s.Kind, Group = s.Group,
                Shape = new Shape([Dim.Of(slots + spareStateRows), .. s.Shape.Dims]),
            };
            return states[s] = d;
        }

        foreach (var n in g.Nodes)
        {
            var ins = n.Inputs.Select(Map).ToList();
            var attrs = n.Attributes;
            var writes = n.Writes.Select(Map).ToList();
            switch (n.Op.Name)
            {
                case "CacheWrite":
                    ins.Add(rows);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "CachedAttention" or "IndexerTopK":
                    ins.Add(n.Op.Name == "CachedAttention" && !attrs.Get("causal", true) ? runEndRows! : rows);
                    ins.Add(runs);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "WindowPool":
                    ins.Add(rows);
                    ins.Add(runs);
                    attrs = attrs.With("slots", (long)slots);
                    break;
                case "CausalConv1d" or "GatedDeltaRule" or "SelectiveScan" or "Wkv7" or "Wkv6":
                    ins.Add(runs);
                    if (snapshots is not null)
                    {
                        ins.Add(snapshots);
                        attrs = attrs.With(RecurrentSnapshots.Attribute, true);
                    }

                    if (commits is not null)
                    {
                        var records = b.State(RecordState(n, ins, slots, rollbackRows));
                        ins.Add(commits);
                        ins.Add(records);
                        writes.Add(records);
                        attrs = attrs.With(RecurrentRollback.RowsAttribute, (long)rollbackRows);
                    }

                    break;
            }

            if (pages is not null && n.Op.Name is "CacheWrite" or "CachedAttention" or "IndexerTopK")
            {
                ins.Add(pages);
                attrs = attrs.With("block_size", (long)blockSize);
            }

            IReadOnlyList<Value> outs;
            using (b.Group(n.Group))
            {
                outs = b.Emit(n.Op, ins, attrs, n.Outputs.Length == 1 ? n.Outputs[0].Name : null, writes);
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

        return new Result(b.Build(), states, slots)
        {
            BlockSize = blockSize,
            BlocksPerSlot = blockSize > 0 ? blocksPerSlot : 0,
            RollbackRows = rollback ? rollbackRows : 0,
            StepSnapshots = stepSnapshots,
        };
    }

    /// <summary>
    /// Whether the graph has a recurrent op and every op that uses per-slot state other than a KV cache is one with a snapshot form
    /// (<see cref="RecurrentSnapshots"/>; e.g. not the compressor ring of <c>WindowPool</c>).
    /// </summary>
    private static bool HasStepSnapshotForm(ComputeGraph g)
    {
        bool any = false;
        foreach (var n in g.Nodes)
        {
            bool recurrent = RecurrentRollback.Ops.Contains(n.Op.Name);
            any |= recurrent;
            if (!recurrent && n.Inputs.Concat(n.Writes).Any(v => v.Kind == ValueKind.State && !IsKvState(v.State!)))
            {
                return false;
            }
        }

        return any;
    }

    /// <summary>
    /// A compressed KV cache (<see cref="StateDefinition.Compression"/> r) of a multi-sequence graph: S equal slot ranges of ceil(Ctx / (S·r)) entries,
    /// or, paged (block size B, compiled with Ctx = pool rows), ceil(B / r) rows per block of the pool.
    /// </summary>
    private static StateDefinition CompressedState(StateDefinition s, int slots, int blockSize)
    {
        var context = Dim.Sym(WellKnown.ContextSymbol);
        long r = s.Compression;
        var rows = blockSize > 0 ? context.CeilDiv(blockSize) * Dim.Of((blockSize + r - 1) / r) : context.CeilDiv(slots * r) * Dim.Of(slots);
        return new StateDefinition
        {
            Name = s.Name, Type = s.Type, Kind = s.Kind, Group = s.Group, Compression = s.Compression,
            Shape = new Shape([rows, .. s.Shape.Dims.Skip(1)]),
        };
    }

    /// <summary>The record state [S, banks, K, W] of a recurrent op in rollback form, named after the op's state.</summary>
    private static StateDefinition RecordState(Node n, IReadOnlyList<Value> inputs, int slots, int rollbackRows)
    {
        var state = n.Inputs.First(v => v.Kind == ValueKind.State).State!;
        long width = RecurrentRollback.RecordWidth(n.Op.Name, inputs);
        return new StateDefinition
        {
            Name = state.Name + RecurrentRollback.RecordStateSuffix,
            Type = DataType.F32,
            Kind = StateKind.Recurrent,
            Group = state.Group,
            Shape = new Shape(Dim.Of(slots), Dim.Of(RecurrentRollback.Banks), Dim.Of(rollbackRows), Dim.Of(width)),
        };
    }
}
