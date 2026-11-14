namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// The dimensions of a model that size its KV cache and activations, read before it is downloaded (a GGUF header or a checkpoint's
/// configuration). Every value is null when the source does not state it.
/// </summary>
/// <param name="Layers">Transformer blocks.</param>
/// <param name="Heads">Attention (query) heads.</param>
/// <param name="KvHeads">Key/value heads.</param>
/// <param name="HiddenSize">Hidden size.</param>
/// <param name="KeyLength">Rows of a key head.</param>
/// <param name="ValueLength">Rows of a value head.</param>
/// <param name="ContextLength">Context length the model was trained for.</param>
/// <param name="VocabularySize">Tokens in the vocabulary.</param>
/// <param name="LatentRank">Rank of a compressed (latent) KV cache, else null.</param>
/// <param name="RopeDimension">Rotary dimensions cached next to the latent.</param>
internal sealed record ModelShape(
    int? Layers,
    int? Heads,
    int? KvHeads,
    int? HiddenSize,
    int? KeyLength,
    int? ValueLength,
    int? ContextLength,
    int? VocabularySize,
    int? LatentRank = null,
    int? RopeDimension = null)
{
    /// <summary>KV cache elements per token and layer, when the shape states enough to tell.</summary>
    public long? KvElementsPerTokenLayer
    {
        get
        {
            if (LatentRank is > 0)
            {
                return LatentRank.Value + (RopeDimension ?? 0);
            }

            int? key = KeyLength ?? (HiddenSize is > 0 && Heads is > 0 ? HiddenSize / Heads : null);
            if (KvHeads is not > 0 || key is not > 0)
            {
                return null;
            }

            return (long)KvHeads.Value * (key.Value + (ValueLength ?? key.Value));
        }
    }
}
