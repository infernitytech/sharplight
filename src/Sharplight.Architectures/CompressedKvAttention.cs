using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Attention over a sliding window of uncompressed keys plus a compressed cache of pooled windows (DeepSeek-V4's hybrid attention) as a reusable
/// block. One key-value head of D dimensions serves every query head and is its own value (keys = values); a query attends the keys of its last
/// <see cref="CompressedKvAttentionOptions.Window"/> positions and, in the same softmax, the compressed entries of the complete windows of r
/// positions before it: every entry, or with an indexer the <see cref="CompressedKvAttentionOptions.TopK"/> entries it selects.
/// </summary>
/// <remarks>
/// Weights below the layer prefix: the compressed query <c>attention.q_down.weight</c> [q_rank, hidden], <c>attention.q_down_norm.weight</c>,
/// <c>attention.q_up.weight</c> [H·D, q_rank] (every head is then RMS-normalized without a weight); the key <c>attention.kv.weight</c> [D, hidden]
/// with <c>attention.kv_norm.weight</c>; <c>attention.sinks</c> (f32 [H]); the grouped output <c>attention.o_group.weight</c> [G, R, H·D / G]
/// (group g maps its heads' values to R channels) and <c>attention.o.weight</c> [hidden, G·R]. The rotary dimensions are the last ones of every
/// query and key, rotated at the row's position; the attended values are rotated back at the query's position before the output projection.
/// Compression (ratio r &gt; 0): the compressor <c>attention.compressor.{kv,gate}.weight</c> [C, hidden], <c>attention.compressor.position_bias</c>
/// [r, C] and <c>attention.compressor.norm.weight</c> [D] pools every window (<c>WindowPool</c>; C = 2·D: overlapping windows), RMS-normalizes
/// the entry and rotates it at the window's first position. Indexer (<c>attention.indexer.q_up.weight</c> [Hi·Di, q_rank],
/// <c>attention.indexer.head_weights.weight</c> [Hi, hidden] and a compressor of its own below <c>attention.indexer.compressor.</c>): the score
/// of entry e for row t is <c>Σ_h w[t, h] · relu(q[t, h] · key[e])</c> with w scaled by 1/sqrt(Hi·Di), and the best top-k entries are attended
/// (<c>IndexerTopK</c>).
/// </remarks>
internal sealed class CompressedKvAttention(CompressedKvAttentionOptions options, int heads, int ropeDims, float eps)
{
    /// <summary>The settings.</summary>
    public CompressedKvAttentionOptions Options => options;

    /// <summary>The states of layer <paramref name="il"/> below <paramref name="prefix"/>.</summary>
    public CompressedKvLayerStates CreateStates(Func<string, WeightInfo?> weights, string prefix, string group, int il, DataType kvType)
    {
        int dims = Rows(weights, prefix + "attention.kv.weight");
        var keys = new StateDefinition
        {
            Name = $"{group}.kv_cache",
            Type = AdapterOptions.KvRowType(kvType, dims),
            Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, dims),
            Kind = StateKind.KeyCache,
            Group = group,
        };
        int ratio = options.Ratios[il];
        if (ratio == 0)
        {
            return new CompressedKvLayerStates(keys, null, null, null, null);
        }

        var entryType = kvType == DataType.F32 ? DataType.F32 : DataType.F16;
        var (entries, pool) = CompressorStates(weights, prefix + "attention.compressor.", $"{group}.compressed", group, ratio, entryType);
        if (weights(prefix + "attention.indexer.q_up.weight") is null)
        {
            return new CompressedKvLayerStates(keys, entries, pool, null, null);
        }

        var (indexEntries, indexPool) = CompressorStates(weights, prefix + "attention.indexer.compressor.", $"{group}.index_compressed", group, ratio, entryType);
        return new CompressedKvLayerStates(keys, entries, pool, indexEntries, indexPool);
    }

    /// <summary>Emits the attention of one layer and returns its output [T, hidden].</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="h">Normalized layer input [T, hidden].</param>
    /// <param name="prefix">Weight name prefix of the layer.</param>
    /// <param name="il">The layer index (its compression ratio).</param>
    /// <param name="states">The layer's states (<see cref="CreateStates"/>).</param>
    /// <param name="positions">Positions [T, 1] of the rows.</param>
    /// <param name="groupIds">Group indices [T, G] (<see cref="LatentAttention.HeadIndices"/> with G groups).</param>
    /// <param name="rotate">The layer's rotary embedding of a [T, n, rope] tensor at the given positions [T, 1].</param>
    public Value Build(GraphBuilder b, Value h, string prefix, int il, CompressedKvLayerStates states, Value positions, Value groupIds,
        Func<Value, Value, Value> rotate)
    {
        int dims = (int)states.Keys.Shape[2].Fixed;
        int ratio = options.Ratios[il];
        var queryInput = b.RmsNorm(b.Linear(h, prefix + "attention.q_down.weight"), b.Weight(prefix + "attention.q_down_norm.weight"), eps);
        var q = b.RmsNorm(b.Reshape(b.Linear(queryInput, prefix + "attention.q_up.weight", name: $"{prefix}q"), 0, heads, dims), null, eps);
        q = RotateTail(b, q, dims, positions, rotate);

        var key = b.RmsNorm(b.Linear(h, prefix + "attention.kv.weight"), b.Weight(prefix + "attention.kv_norm.weight"), eps, $"{prefix}kv");
        var keys = b.State(states.Keys);
        b.CacheWrite(keys, RotateTail(b, b.Reshape(key, 0, 1, dims), dims, positions, rotate));

        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("scale", 1f / MathF.Sqrt(dims)), ("offset", WellKnown.PastScalar), ("window", (long)options.Window), ("sinks", true));
        var inputs = new List<Value> { q, keys, keys, b.Weight(prefix + "attention.sinks") };
        if (ratio > 0)
        {
            var entries = Compress(b, h, prefix + "attention.compressor.", states.Entries!, states.Pool!, ratio, positions, rotate);
            inputs.Add(entries);
            attributes = attributes.With("compression", (long)ratio);
            if (states.IndexEntries is not null)
            {
                inputs.Add(Select(b, h, queryInput, prefix, states, ratio, positions, rotate));
                attributes = attributes.With("selection", true);
            }
        }

        var attended = b.Emit1("CachedAttention", inputs, attributes, $"{prefix}attn");                                         // [T, H, D]
        // Keys are values, so the attended values carry the keys' rotation; it is undone at the query's position.
        attended = RotateTail(b, attended, dims, Shift(b, positions, -1f, 0f), rotate);
        int groups = options.OutputGroups;
        var grouped = b.Reshape(attended, 0, groups, (long)heads * dims / groups);
        var projected = b.Emit1("MatMulId", [grouped, b.Weight(prefix + "attention.o_group.weight"), groupIds], null, $"{prefix}attn_groups");   // [T, G, R]
        long rank = b.Weight(prefix + "attention.o_group.weight").Shape[1].Fixed;
        return b.Linear(b.Reshape(projected, 0, groups * rank), prefix + "attention.o.weight", name: $"{prefix}attn_out");
    }

    // The compressed entries of a compressor below p: pooled windows, RMS-normalized and rotated at the window's first position, written to the
    // compressed cache when a row closes its window. Returns the cache state.
    private Value Compress(GraphBuilder b, Value h, string p, StateDefinition entries, StateDefinition pool, int ratio, Value positions,
        Func<Value, Value, Value> rotate)
    {
        int dims = (int)entries.Shape[2].Fixed;
        var values = b.Linear(h, p + "kv.weight");
        var scores = b.Linear(h, p + "gate.weight");
        bool overlap = values.Shape[1].Fixed == 2L * dims;
        var ring = b.State(pool);
        var pooled = b.Emit("WindowPool", [values, scores, b.Weight(p + "position_bias"), positions, ring],
            Attributes.From(("ratio", (long)ratio), ("overlap", overlap)), writes: [ring])[0];                                     // [T, D]
        var entry = b.Reshape(b.RmsNorm(pooled, b.Weight(p + "norm.weight"), eps), 0, 1, dims);
        entry = RotateTail(b, entry, dims, Shift(b, positions, 1f, 1 - ratio), rotate);
        var cache = b.State(entries);
        b.RuntimeScalar(WellKnown.PastScalar);
        b.Emit("CacheWrite", [cache, entry], Attributes.From(("offset", WellKnown.PastScalar), ("compression", (long)ratio)), writes: [cache]);
        return cache;
    }

    // The indexer's selection of compressed entries i32 [T, top_k].
    private Value Select(GraphBuilder b, Value h, Value queryInput, string prefix, CompressedKvLayerStates states, int ratio, Value positions,
        Func<Value, Value, Value> rotate)
    {
        string p = prefix + "attention.indexer.";
        int dims = (int)states.IndexEntries!.Shape[2].Fixed;
        int indexHeads = (int)b.Weight(p + "head_weights.weight").Shape[0].Fixed;
        var keys = Compress(b, h, p + "compressor.", states.IndexEntries, states.IndexPool!, ratio, positions, rotate);
        var q = RotateTail(b, b.Reshape(b.Linear(queryInput, p + "q_up.weight", name: $"{prefix}indexer_q"), 0, indexHeads, dims), dims, positions, rotate);
        var weights = b.Scale(b.Linear(h, p + "head_weights.weight"), 1f / MathF.Sqrt((float)indexHeads * dims), name: $"{prefix}indexer_w");
        var attributes = Attributes.From(("k", (long)options.TopK), ("offset", WellKnown.PastScalar), ("compression", (long)ratio));
        return b.Emit1("IndexerTopK", [q, keys, weights], attributes, $"{prefix}indexer_topk");
    }

    // Rotates the last rope dimensions of x [T, n, dims] at the given positions.
    private Value RotateTail(GraphBuilder b, Value x, int dims, Value positions, Func<Value, Value, Value> rotate)
        => b.Concat(2, b.Slice(x, 2, 0, dims - ropeDims), rotate(b.Slice(x, 2, dims - ropeDims, ropeDims), positions));

    // factor · position + offset of every row, as i32 positions [T, 1].
    private static Value Shift(GraphBuilder b, Value positions, float factor, float offset)
    {
        var shifted = b.Scale(b.Emit1("Cast", [positions], Attributes.From(("type", "f32"))), factor, offset);
        return b.Emit1("Cast", [shifted], Attributes.From(("type", "i32")));
    }

    // The compressed cache [ceil(ctx / r), 1, D] and the compressor ring [P, 2C] of the compressor below p.
    private static (StateDefinition Entries, StateDefinition Pool) CompressorStates(Func<string, WeightInfo?> weights, string p, string name, string group,
        int ratio, DataType entryType)
    {
        int channels = Rows(weights, p + "kv.weight");
        int dims = Rows(weights, p + "norm.weight");
        bool overlap = channels == 2 * dims;
        var entries = new StateDefinition
        {
            Name = $"{name}_cache",
            Type = entryType,
            Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol).CeilDiv(ratio), 1, dims),
            Kind = StateKind.KeyCache,
            Group = group,
            Compression = ratio,
        };
        var pool = new StateDefinition
        {
            Name = $"{name}_pool",
            Type = DataType.F32,
            Shape = new Shape((overlap ? 2 : 1) * ratio, 2 * channels),
            Kind = StateKind.Recurrent,
            Group = group,
        };
        return (entries, pool);
    }

    private static int Rows(Func<string, WeightInfo?> weights, string name)
        => (int)(weights(name) ?? throw new InvalidDataException($"latent_decoder: {name} is missing.")).Shape[0].Fixed;
}
