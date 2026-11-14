namespace Sharplight.Gguf;

/// <summary>
/// The dimensions of a GGUF model that size its KV cache, read from the start of the file (see <see cref="GgufHeaderProbe"/>).
/// Every value is null when the part of the header that was read does not contain it.
/// </summary>
/// <param name="Architecture">The <c>general.architecture</c> name.</param>
/// <param name="BlockCount">Transformer blocks (layers).</param>
/// <param name="HeadCount">Attention (query) heads; the largest when the count varies per layer.</param>
/// <param name="KvHeadCount">Key/value heads; the largest when the count varies per layer.</param>
/// <param name="EmbeddingLength">Hidden size.</param>
/// <param name="KeyLength">Rows of a key head (default: hidden size / heads).</param>
/// <param name="ValueLength">Rows of a value head (default: the key length).</param>
/// <param name="ContextLength">Context length the model was trained for.</param>
/// <param name="VocabularySize">Tokens in the vocabulary, when the token list was read.</param>
/// <param name="KvLoraRank">Rank of the compressed (latent) KV cache of multi-head latent attention, else null.</param>
/// <param name="RopeDimension">Rotary dimensions per head (cached next to the latent of latent attention).</param>
public sealed record GgufModelShape(
    string? Architecture,
    int? BlockCount,
    int? HeadCount,
    int? KvHeadCount,
    int? EmbeddingLength,
    int? KeyLength,
    int? ValueLength,
    int? ContextLength,
    int? VocabularySize,
    int? KvLoraRank = null,
    int? RopeDimension = null);
