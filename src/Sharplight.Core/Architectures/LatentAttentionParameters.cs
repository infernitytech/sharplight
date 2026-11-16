namespace Sharplight.Architectures;

/// <summary>Dimensions and scales of one multi-head latent attention layer (see <see cref="LatentAttention"/>).</summary>
/// <param name="Heads">Attention heads.</param>
/// <param name="QueryRank">Rank of the low-rank query compression (0: the query is projected directly from the layer input).</param>
/// <param name="LatentRank">Rank of the compressed key/value latent (kv_lora_rank); the value part of every cache row.</param>
/// <param name="NopeDims">Non-rotary query/key dimensions per head.</param>
/// <param name="RopeDims">Rotary query/key dimensions per head (one rotary key shared by all heads).</param>
/// <param name="ValueDims">Value dimensions per head.</param>
/// <param name="Scale">Softmax scale of the scores (1/sqrt(nope + rope) times any yarn magnitude correction).</param>
/// <param name="RopeMagnitude">Factor on the rotated query/key parts (yarn cos/sin magnitude; 1 for most checkpoints).</param>
/// <param name="Eps">Epsilon of the RMS norms of the compressed query and latent.</param>
/// <param name="Window">Sliding window: a query attends only its last <paramref name="Window"/> positions (0 = every visible position).</param>
internal sealed record LatentAttentionParameters(
    int Heads, int QueryRank, int LatentRank, int NopeDims, int RopeDims, int ValueDims, float Scale, float RopeMagnitude, float Eps, int Window = 0)
{
    /// <summary>Width of one cache row: the latent followed by the shared rotary key.</summary>
    public int CacheRowDims => LatentRank + RopeDims;
}
