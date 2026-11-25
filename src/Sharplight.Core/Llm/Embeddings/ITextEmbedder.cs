using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Text embeddings (kind <see cref="Models.ModelKinds.TextEmbedding"/>): texts in, one vector per text out. Encoder embedders (BERT
/// family: bge, e5, MiniLM, nomic-embed, XLM-R models) and decoder-based ones (Qwen3-Embedding: causal, last-token pooling); inputs are
/// tokenized, truncated, packed into batches and pooled and normalized as the model declares (pooling and normalization can be
/// overridden at load with <see cref="Models.AdapterOptions.EmbeddingPooling"/> and <see cref="Models.AdapterOptions.EmbeddingNormalize"/>).
/// Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>); requests are serialized.
/// </summary>
/// <example>
/// <code>
/// services.AddTextEmbedder(o => { o.ModelPath = "bge-small-en-v1.5.slm"; o.Devices = [slot]; });
/// var embedder = provider.GetRequiredService&lt;ITextEmbedder&gt;();
///
/// // A batch: one vector per text, in input order, plus token counts.
/// EmbeddingResult result = await embedder.EmbedAsync(["first text", "second text"]);
///
/// // Models that distinguish queries from documents define named prompts.
/// var query = await embedder.EmbedAsync(["what is a vector?"], new TextEmbeddingOptions { Prompt = "query" });
/// </code>
/// </example>
public interface ITextEmbedder : ITaskClient
{
    /// <summary>Length of the vectors (before an <see cref="TextEmbeddingOptions.Dimensions"/> cut; per token for pooling "none").</summary>
    public int Dimensions { get; }

    /// <summary>Longest input in tokens, including special tokens: the model's limit, at most <see cref="TaskLoadOptions.MaxBatchTokens"/>.</summary>
    public int MaxSequenceLength { get; }

    /// <summary>Named prompts (e.g. "query", "document") selectable with <see cref="TextEmbeddingOptions.Prompt"/>.</summary>
    public IReadOnlyDictionary<string, string> Prompts { get; }

    /// <summary>Token ids of one input as the model sees it: prompt prefix, special tokens, truncation. Runs on the calling thread (no graph).</summary>
    /// <param name="text">The input text.</param>
    /// <param name="options">The prompt and truncation, or <see langword="null"/> for the defaults.</param>
    /// <returns>The token ids.</returns>
    public int[] Tokenize(string text, TextEmbeddingOptions? options = null);

    /// <summary>Embeds texts (batched); vectors are in input order.</summary>
    /// <param name="texts">The inputs.</param>
    /// <param name="options">Prompt, truncation and dimensions, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    /// <returns>One vector per input and the token counts.</returns>
    public Task<EmbeddingResult> EmbedAsync(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Embeds pre-tokenized inputs as given (no prompt and no special tokens are added; truncated to <see cref="MaxSequenceLength"/>).</summary>
    /// <param name="tokens">The token ids of each input.</param>
    /// <param name="options">Truncation and dimensions (prompts do not apply), or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    /// <returns>One vector per input and the token counts.</returns>
    public Task<EmbeddingResult> EmbedTokensAsync(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default);
}
