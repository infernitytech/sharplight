using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// A block-sparse attention indexer (MiniMax-M3 sparse attention) as a reusable block: per query and per selection head (one per key-value
/// group), the key blocks of <see cref="BlockSparseIndexerOptions.BlockSize"/> positions are scored by their best key and the layer's attention
/// only attends to the best <see cref="BlockSparseIndexerOptions.Blocks"/> of them, always including the query's latest
/// <see cref="BlockSparseIndexerOptions.LocalBlocks"/> (<c>CachedAttention</c> with a per-head selection). Its keys live in their own cache of
/// one row per position.
/// </summary>
/// <remarks>
/// Weights below the layer prefix: <c>attention.block_indexer.q.weight</c> [Hs·D, hidden], <c>attention.block_indexer.k.weight</c> [D, hidden]
/// and the RMS norm weights <c>attention.block_indexer.{q,k}_norm.weight</c> [D]. Score of key position j for query row t and head s:
/// q[t, s] · k[j] with q = RMSNorm(q-projection of the head), k = RMSNorm(k-projection), both rotated like the layer's attention. The selection
/// is <c>IndexerTopK</c> with per-head groups; rows that see at most the budget keep every position.
/// </remarks>
internal sealed class BlockSparseIndexer(BlockSparseIndexerOptions options, float normEps)
{
    private const string Prefix = "attention.block_indexer.";

    /// <summary>The settings.</summary>
    public BlockSparseIndexerOptions Options => options;

    /// <summary>Whether the layer below <paramref name="prefix"/> has a block indexer.</summary>
    public static bool IsPresent(Func<string, WeightInfo?> weights, string prefix) => weights(prefix + Prefix + "k.weight") is not null;

    /// <summary>
    /// The key cache of the layer below <paramref name="prefix"/>: [ctx, 1, D] in f32 when the KV cache is f32, otherwise f16 (block-quantized
    /// KV caches keep the indexer keys in f16).
    /// </summary>
    public static StateDefinition CreateKeyCache(Func<string, WeightInfo?> weights, string prefix, string group, DataType kvType) => new()
    {
        Name = $"{group}.block_indexer_key_cache",
        Type = kvType == DataType.F32 ? DataType.F32 : DataType.F16,
        Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, weights(prefix + Prefix + "k.weight")!.Shape[0]),
        Kind = StateKind.KeyCache,
        Group = group,
    };

    /// <summary>Emits the indexer of one layer: writes its key cache rows and returns the selected positions i32 [T, Hs, budget].</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="h">Normalized layer input [T, hidden].</param>
    /// <param name="prefix">Weight name prefix of the layer.</param>
    /// <param name="keyCache">The layer's key cache (<see cref="CreateKeyCache"/>).</param>
    /// <param name="rotate">The layer's rotary embedding of a [T, heads, D] tensor.</param>
    public Value Select(GraphBuilder b, Value h, string prefix, StateDefinition keyCache, Func<Value, Value> rotate)
    {
        int dims = (int)b.Weight(prefix + Prefix + "k.weight").Shape[0].Fixed;
        int heads = (int)(b.Weight(prefix + Prefix + "q.weight").Shape[0].Fixed / dims);
        var q = b.Reshape(b.Linear(h, prefix + Prefix + "q.weight", name: $"{prefix}block_indexer_q"), 0, heads, dims);
        var k = b.Reshape(b.Linear(h, prefix + Prefix + "k.weight", name: $"{prefix}block_indexer_k"), 0, 1, dims);
        q = rotate(b.RmsNorm(q, b.Weight(prefix + Prefix + "q_norm.weight"), normEps));
        k = rotate(b.RmsNorm(k, b.Weight(prefix + Prefix + "k_norm.weight"), normEps));
        var state = b.State(keyCache);
        b.CacheWrite(state, k);
        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("k", (long)options.TokenBudget), ("offset", WellKnown.PastScalar), ("group", (long)options.BlockSize),
            ("local", (long)options.LocalBlocks), ("per_head", true));
        return b.Emit1("IndexerTopK", [q, state], attributes, $"{prefix}block_indexer_topk");
    }
}
