using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// The load settings text embedders and rerankers start from (<c>AddTextEmbedder</c>, <c>AddTextReranker</c>, the hosting loader,
/// package precompilation): batches of up to <see cref="MaxBatchTokens"/> tokens, which also bounds the longest input, and F32 matrices
/// stored as F16 on GPUs (<see cref="TaskLoadOptions.HalfPrecisionMatrices"/> = null).
/// </summary>
public static class TextEmbeddingLoadDefaults
{
    /// <summary>Most tokens of all sequences of a batch together (and so the longest input).</summary>
    public const int MaxBatchTokens = 8192;

    /// <summary>Sets <paramref name="options"/> to the defaults (call before applying caller settings).</summary>
    /// <param name="options">The options to set.</param>
    public static void Apply(TaskLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.MaxBatchTokens = MaxBatchTokens;
        options.HalfPrecisionMatrices = null;
    }
}
