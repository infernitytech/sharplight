using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The lightning indexer of DeepSeek sparse attention (DSA: DeepSeek-V3.2, GLM-5, Hunyuan v4) as a reusable block: a few small scoring heads
/// pick <see cref="SparseAttentionIndexerOptions.TokenBudget"/> key positions per query, and the layer's attention only attends to them
/// (<c>CachedAttention</c> with a selection). Its keys live in their own cache of one row per position.
/// </summary>
/// <remarks>
/// Per query row t and key position j: <c>score = Σ_h w[t, h] · relu(q[t, h] · k[j])</c> with q = the query input projected by
/// <c>attention.indexer.q_up.weight</c> [H·D, q_in] (DSA: the normalized compressed query), k = LayerNorm(<c>attention.indexer.k.weight</c>
/// [D, hidden] · h) with <c>attention.indexer.k_norm.{weight,bias}</c>, and w = <c>attention.indexer.head_weights.weight</c> [H, hidden] · h
/// scaled by 1/sqrt(H·D) (the reference's per-head and softmax scales; positive factors commute with the relu). The rotary part of q and k
/// (the layer's rope dimensions, at the start or at the end of each row) is rotated like the attention's, with the indexer's pairing style.
/// The selection is <c>IndexerTopK</c>; rows that see at most the budget keep every position.
/// Mistral Large 4: the query input is LayerNorm(h) with <c>attention.indexer.q_norm.{weight,bias}</c> instead of the attention's query input,
/// and <c>attention.indexer.ssmax.{weight,bias}</c> scale the indexer heads (<see cref="ScalableSoftmax"/>, before the relu).
/// Pooled keys (<see cref="SparseAttentionIndexerOptions.PoolSize"/> = G &gt; 1, GLM-5.3): the cache rows hold the key and a gate
/// <c>attention.indexer.pool_gate.weight</c> [D, hidden] · h; the indexer scores one key per G consecutive positions (per channel the softmax
/// over the members of gate + <c>attention.indexer.pool_position.weight</c> [G, D] weights their keys) and picks budget / G whole pools,
/// optionally followed by the query's incomplete pool.
/// </remarks>
internal sealed class SparseAttentionIndexer(SparseAttentionIndexerOptions options)
{
    /// <summary>The settings.</summary>
    public SparseAttentionIndexerOptions Options => options;

    /// <summary>Whether the layer below <paramref name="prefix"/> has its own indexer (other indexed layers reuse the previous selection).</summary>
    public static bool IsPresent(Func<string, WeightInfo?> weights, string prefix) => weights(prefix + "attention.indexer.k.weight") is not null;

    /// <summary>
    /// The key cache of the layer below <paramref name="prefix"/>: [ctx, 1, D] (pooled keys: [ctx, 1, 2·D], key and gate) in f32 when the KV cache
    /// is f32, otherwise f16 (block-quantized KV caches keep the indexer keys in f16).
    /// </summary>
    public StateDefinition CreateKeyCache(Func<string, WeightInfo?> weights, string prefix, string group, DataType kvType) => new()
    {
        Name = $"{group}.indexer_key_cache",
        Type = kvType == DataType.F32 ? DataType.F32 : DataType.F16,
        Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, weights(prefix + "attention.indexer.k.weight")!.Shape[0].Fixed * (options.Pooled ? 2 : 1)),
        Kind = StateKind.KeyCache,
        Group = group,
    };

    /// <summary>Emits the indexer of one layer: writes its key cache rows and returns the selected positions i32 [T, budget].</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="h">Normalized layer input [T, hidden].</param>
    /// <param name="queryInput">Input of the indexer's query projection [T, q_in] (DSA: the normalized compressed query).</param>
    /// <param name="prefix">Weight name prefix of the layer.</param>
    /// <param name="keyCache">The layer's key cache (<see cref="CreateKeyCache"/>).</param>
    /// <param name="ropeDims">Rotary dimensions of each query and key.</param>
    /// <param name="rotate">Applies the indexer's rotary embedding to a [T, heads, <paramref name="ropeDims"/>] tensor.</param>
    /// <param name="logPositions">ln(pos + 1) of every row (<see cref="ScalableSoftmax.LogPositions"/>) when the model has SSMax factors.</param>
    public Value Select(GraphBuilder b, Value h, Value queryInput, string prefix, StateDefinition keyCache, int ropeDims, Func<Value, Value> rotate,
        Value? logPositions = null)
    {
        int heads = (int)b.Weight(prefix + "attention.indexer.head_weights.weight").Shape[0].Fixed;
        int dims = (int)b.Weight(prefix + "attention.indexer.k.weight").Shape[0].Fixed;
        if (b.HasWeight(prefix + "attention.indexer.q_norm.weight"))
        {
            queryInput = b.Emit1("LayerNorm", [h, b.Weight(prefix + "attention.indexer.q_norm.weight"), b.Weight(prefix + "attention.indexer.q_norm.bias")],
                Attributes.From(("eps", options.NormEps)), $"{prefix}indexer_q_norm");
        }

        var q = b.Reshape(b.Linear(queryInput, prefix + "attention.indexer.q_up.weight", name: $"{prefix}indexer_q"), 0, heads, dims);
        q = ScalableSoftmax.Apply(b, q, logPositions, prefix + "attention.indexer.ssmax");
        var k = b.Emit1("LayerNorm", [b.Linear(h, prefix + "attention.indexer.k.weight"), b.Weight(prefix + "attention.indexer.k_norm.weight"),
            b.Weight(prefix + "attention.indexer.k_norm.bias")], Attributes.From(("eps", options.NormEps)), $"{prefix}indexer_k");
        q = Rotate(b, q, dims, ropeDims, rotate);
        k = Rotate(b, b.Reshape(k, 0, 1, dims), dims, ropeDims, rotate);
        if (options.Pooled)
        {
            k = b.Concat(2, k, b.Reshape(b.Linear(h, prefix + "attention.indexer.pool_gate.weight", name: $"{prefix}indexer_gate"), 0, 1, dims));
        }

        var state = b.State(keyCache);
        b.CacheWrite(state, k);

        var weights = b.Scale(b.Linear(h, prefix + "attention.indexer.head_weights.weight"), 1f / MathF.Sqrt(heads * dims), name: $"{prefix}indexer_w");
        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("k", (long)options.TokenBudget), ("offset", WellKnown.PastScalar));
        Value[] inputs = [q, state, weights];
        if (options.Pooled)
        {
            attributes = attributes.With("group", (long)options.PoolSize).With("group_score", "pooled").With("tail", options.PoolTail);
            inputs = [.. inputs, b.Weight(prefix + "attention.indexer.pool_position.weight")];
        }

        return b.Emit1("IndexerTopK", inputs, attributes, $"{prefix}indexer_topk");
    }

    // The rotary part is the start (DeepSeek-V3.2, GLM-5) or the end (Hunyuan v4) of every row; GLM-5.3 has none.
    private Value Rotate(GraphBuilder b, Value x, int dims, int ropeDims, Func<Value, Value> rotate)
    {
        if (ropeDims == 0)
        {
            return x;
        }

        int start = options.RopeLast ? dims - ropeDims : 0;
        var rotated = rotate(b.Slice(x, 2, start, ropeDims));
        var pass = b.Slice(x, 2, options.RopeLast ? 0 : ropeDims, dims - ropeDims);
        return options.RopeLast ? b.Concat(2, pass, rotated) : b.Concat(2, rotated, pass);
    }
}
