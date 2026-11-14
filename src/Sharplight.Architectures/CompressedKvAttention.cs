using System.Runtime.InteropServices;
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
/// <c>attention.q_up.weight</c> [H·D, q_rank] (with <see cref="CompressedKvAttentionOptions.QueryNorm"/> every head is then RMS-normalized
/// without a weight); the key <c>attention.kv.weight</c> [D, hidden] with <c>attention.kv_norm.weight</c>; <c>attention.sinks</c> (f32 [H]); the
/// grouped output <c>attention.o_group.weight</c> [G, R, H·D / G] (group g maps its heads' values to R channels) and <c>attention.o.weight</c>
/// [hidden, G·R]. The rotary dimensions are the last ones of every query and key, rotated at the row's position; the attended values are rotated
/// back at the query's position before the output projection.
/// Compression (ratio r &gt; 0): the compressor <c>attention.compressor.{kv,gate}.weight</c> [C, hidden], the optional
/// <c>attention.compressor.position_bias</c> [r, C] and <c>attention.compressor.norm.weight</c> [D] pools every window (<c>WindowPool</c>;
/// C = 2·D: overlapping windows; r = 1 and no gate: every position is its own entry), RMS-normalizes the entry and rotates it at the window's
/// first position. Indexer (<c>attention.indexer.q_up.weight</c> [Hi·Di, q_rank], <c>attention.indexer.head_weights.weight</c> [Hi, hidden]):
/// the score of entry e for row t is <c>Σ_h w[t, h] · relu(q[t, h] · key[e])</c> with w scaled by 1/sqrt(Hi·Di), and the best top-k entries are
/// attended (<c>IndexerTopK</c>). The index keys come from a compressor of the indexer's own below <c>attention.indexer.compressor.</c>, or
/// (DeepSeek-V4.1) from every entry before its rotation: <c>attention.indexer.k.weight</c> [Di, D] and <c>attention.indexer.k_norm.weight</c>,
/// rotated like the entry. Layers can attend an earlier layer's compressed caches and reuse its selection
/// (<see cref="CompressedKvAttentionOptions.KvSources"/>, <see cref="CompressedKvAttentionOptions.IndexSources"/>), and the candidate source
/// layer's indexer can pick candidate blocks inside which the later indexers select.
/// </remarks>
internal sealed class CompressedKvAttention(CompressedKvAttentionOptions options, int heads, int ropeDims, float eps)
{
    /// <summary>The settings.</summary>
    public CompressedKvAttentionOptions Options => options;

    /// <summary>The states of layer <paramref name="il"/> below <paramref name="prefix"/> (the compressed caches of its kv source when that is an earlier layer).</summary>
    public CompressedKvLayerStates CreateStates(Func<string, WeightInfo?> weights, string prefix, string group, int il, DataType kvType,
        IReadOnlyList<CompressedKvLayerStates?> earlier)
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

        int source = options.KvSources[il];
        if (source != il)
        {
            var owner = earlier[source] ?? throw new InvalidDataException($"latent_decoder: layer {il} attends the compressed caches of layer {source}, which has none.");
            return new CompressedKvLayerStates(keys, owner.Entries, null, owner.IndexEntries, null, Borrowed: true);
        }

        var entryType = kvType == DataType.F32 ? DataType.F32 : DataType.F16;
        var (entries, pool) = CompressorStates(weights, prefix + "attention.compressor.", $"{group}.compressed", group, ratio, entryType);
        if (weights(prefix + "attention.indexer.k.weight") is { } indexKey)
        {
            // Index keys derived from the entries: one per entry, no compressor of their own.
            var derived = EntryCache($"{group}.index_compressed_cache", group, ratio, (int)indexKey.Shape[0].Fixed, entryType);
            return new CompressedKvLayerStates(keys, entries, pool, derived, null);
        }

        if (weights(prefix + "attention.indexer.compressor.kv.weight") is null)
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
    /// <param name="shared">The selections and candidate blocks of the earlier layers of the graph (this layer adds its own).</param>
    public Value Build(GraphBuilder b, Value h, string prefix, int il, CompressedKvLayerStates states, Value positions, Value groupIds,
        Func<Value, Value, Value> rotate, CompressedKvShared shared)
    {
        int dims = (int)states.Keys.Shape[2].Fixed;
        int ratio = options.Ratios[il];
        var queryInput = b.RmsNorm(b.Linear(h, prefix + "attention.q_down.weight"), b.Weight(prefix + "attention.q_down_norm.weight"), eps);
        var q = b.Reshape(b.Linear(queryInput, prefix + "attention.q_up.weight", name: $"{prefix}q"), 0, heads, dims);
        if (options.QueryNorm)
        {
            q = b.RmsNorm(q, null, eps);
        }

        q = RotateTail(b, q, dims, positions, rotate);

        var key = b.RmsNorm(b.Linear(h, prefix + "attention.kv.weight"), b.Weight(prefix + "attention.kv_norm.weight"), eps, $"{prefix}kv");
        var keys = b.State(states.Keys);
        b.CacheWrite(keys, RotateTail(b, b.Reshape(key, 0, 1, dims), dims, positions, rotate));

        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("scale", 1f / MathF.Sqrt(dims)), ("offset", WellKnown.PastScalar), ("window", (long)options.Window), ("sinks", true));
        var inputs = new List<Value> { q, keys, keys, b.Weight(prefix + "attention.sinks") };
        if (ratio > 0)
        {
            Value entries;
            if (states.Borrowed)
            {
                entries = b.State(states.Entries!);
            }
            else
            {
                var derived = b.HasWeight(prefix + "attention.indexer.k.weight") ? states.IndexEntries : null;
                entries = Compress(b, h, prefix + "attention.compressor.", states.Entries!, states.Pool, ratio, positions, rotate, derived, prefix);
            }

            inputs.Add(entries);
            attributes = attributes.With("compression", (long)ratio);
            if (Selection(b, h, queryInput, prefix, il, states, ratio, positions, rotate, shared) is { } selection)
            {
                inputs.Add(selection);
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

    // The compressed entries of a compressor below p: pooled windows (a single position's value for one position per entry without a gate),
    // RMS-normalized and rotated at the window's first position, written to the compressed cache when a row closes its window. With
    // indexKeys the index keys derive from the normalized entries (indexer.k below the layer prefix) and go to that cache. Returns the cache.
    private Value Compress(GraphBuilder b, Value h, string p, StateDefinition entries, StateDefinition? pool, int ratio, Value positions,
        Func<Value, Value, Value> rotate, StateDefinition? indexKeys = null, string layerPrefix = "")
    {
        int dims = (int)entries.Shape[2].Fixed;
        var values = b.Linear(h, p + "kv.weight");
        Value pooled = values;
        if (pool is not null)
        {
            var scores = b.Linear(h, p + "gate.weight");
            int channels = (int)values.Shape[1].Fixed;
            bool overlap = channels == 2 * dims;
            var bias = b.TryWeight(p + "position_bias") ?? b.Constant(p + "position_bias_zero", DataType.F32, new Shape(ratio, channels),
                MemoryMarshal.AsBytes(new float[ratio * channels].AsSpan()).ToArray());
            var ring = b.State(pool);
            pooled = b.Emit("WindowPool", [values, scores, bias, positions, ring],
                Attributes.From(("ratio", (long)ratio), ("overlap", overlap)), writes: [ring])[0];                                 // [T, D]
        }

        var latent = b.RmsNorm(pooled, b.Weight(p + "norm.weight"), eps);
        var windowStart = Shift(b, positions, 1f, 1 - ratio);
        var entry = RotateTail(b, b.Reshape(latent, 0, 1, dims), dims, windowStart, rotate);
        var cache = b.State(entries);
        b.RuntimeScalar(WellKnown.PastScalar);
        b.Emit("CacheWrite", [cache, entry], Attributes.From(("offset", WellKnown.PastScalar), ("compression", (long)ratio)), writes: [cache]);
        if (indexKeys is not null)
        {
            int indexDims = (int)indexKeys.Shape[2].Fixed;
            var indexKey = b.RmsNorm(b.Linear(latent, layerPrefix + "attention.indexer.k.weight"), b.Weight(layerPrefix + "attention.indexer.k_norm.weight"), eps);
            indexKey = RotateTail(b, b.Reshape(indexKey, 0, 1, indexDims), indexDims, windowStart, rotate);
            var indexCache = b.State(indexKeys);
            b.Emit("CacheWrite", [indexCache, indexKey], Attributes.From(("offset", WellKnown.PastScalar), ("compression", (long)ratio)), writes: [indexCache]);
        }

        return cache;
    }

    // The entries a layer attends (i32 [T, top_k]): its own indexer's selection (restricted to the candidate blocks of an earlier candidate
    // source), its index source's, or none.
    private Value? Selection(GraphBuilder b, Value h, Value queryInput, string prefix, int il, CompressedKvLayerStates states, int ratio, Value positions,
        Func<Value, Value, Value> rotate, CompressedKvShared shared)
    {
        int source = options.IndexSources[il];
        if (source < 0)
        {
            return null;
        }

        if (source != il)
        {
            return shared.Selections.TryGetValue(source, out var reused) ? reused
                : throw new InvalidDataException($"latent_decoder: layer {il} reuses the selection of layer {source}, which has none.");
        }

        if (states.IndexEntries is null || !b.HasWeight(prefix + "attention.indexer.q_up.weight"))
        {
            return null;
        }

        string p = prefix + "attention.indexer.";
        int dims = (int)states.IndexEntries.Shape[2].Fixed;
        int indexHeads = (int)b.Weight(p + "head_weights.weight").Shape[0].Fixed;
        var keys = states.IndexPool is not null
            ? Compress(b, h, p + "compressor.", states.IndexEntries, states.IndexPool, ratio, positions, rotate)
            : b.State(states.IndexEntries);
        var q = RotateTail(b, b.Reshape(b.Linear(queryInput, p + "q_up.weight", name: $"{prefix}indexer_q"), 0, indexHeads, dims), dims, positions, rotate);
        var weights = b.Scale(b.Linear(h, p + "head_weights.weight"), 1f / MathF.Sqrt((float)indexHeads * dims), name: $"{prefix}indexer_w");
        var attributes = Attributes.From(("k", (long)options.TopK), ("offset", WellKnown.PastScalar), ("compression", (long)ratio));
        Value selection;
        if (options.CandidateSource >= 0 && options.CandidateSource < il && shared.Candidates is { } candidates)
        {
            selection = b.Emit1("IndexerTopK", [q, keys, weights, candidates], attributes.With("candidate_group", (long)options.CandidateBlockSize),
                $"{prefix}indexer_topk");
        }
        else
        {
            selection = b.Emit1("IndexerTopK", [q, keys, weights], attributes, $"{prefix}indexer_topk");
        }

        if (options.CandidateSource == il)
        {
            // The candidate blocks: the best entry of a block scores it, the block of the newest entry is always kept.
            int size = options.CandidateBlockSize;
            shared.Candidates = b.Emit1("IndexerTopK", [q, keys, weights], Attributes.From(("k", (long)options.CandidateBlocks * size),
                ("offset", WellKnown.PastScalar), ("compression", (long)ratio), ("group", (long)size), ("group_score", "max"), ("local", 1L)),
                $"{prefix}indexer_candidates");
        }

        shared.Selections[il] = selection;
        return selection;
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

    // The compressed cache [ceil(ctx / r), 1, D] and the compressor ring [P, 2C] of the compressor below p (no ring for one position per entry
    // without a gate).
    private static (StateDefinition Entries, StateDefinition? Pool) CompressorStates(Func<string, WeightInfo?> weights, string p, string name, string group,
        int ratio, DataType entryType)
    {
        int channels = Rows(weights, p + "kv.weight");
        int dims = Rows(weights, p + "norm.weight");
        bool overlap = channels == 2 * dims;
        var entries = EntryCache($"{name}_cache", group, ratio, dims, entryType);
        if (ratio == 1 && weights(p + "gate.weight") is null)
        {
            return (entries, null);
        }

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

    private static StateDefinition EntryCache(string name, string group, int ratio, int dims, DataType type) => new()
    {
        Name = name,
        Type = type,
        Shape = new Shape(ratio == 1 ? Dim.Sym(WellKnown.ContextSymbol) : Dim.Sym(WellKnown.ContextSymbol).CeilDiv(ratio), 1, dims),
        Kind = StateKind.KeyCache,
        Group = group,
        Compression = ratio,
    };

    private static int Rows(Func<string, WeightInfo?> weights, string name)
        => (int)(weights(name) ?? throw new InvalidDataException($"latent_decoder: {name} is missing.")).Shape[0].Fixed;
}
