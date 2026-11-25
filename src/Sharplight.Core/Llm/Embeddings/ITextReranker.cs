using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Reranking (kind <see cref="Models.ModelKinds.TextReranking"/>): scores how relevant documents are to a query. Cross-encoders (BERT /
/// XLM-RoBERTa classifiers) read "query, document" as one pair; decoder rerankers (Qwen3-Reranker) get a judging prompt and score P(yes)
/// from the last token's "yes" / "no" logits. Documents are batched. Follows the task-client pattern of every model kind
/// (<see cref="ITaskClient"/>); requests are serialized.
/// </summary>
/// <example>
/// <code>
/// services.AddTextReranker(o => { o.ModelPath = "bge-reranker-v2-m3.slm"; o.Devices = [slot]; });
/// var reranker = provider.GetRequiredService&lt;ITextReranker&gt;();
/// RerankResult result = await reranker.RerankAsync("what is a vector?", ["a quantity with direction", "a kind of animal"]);
/// // result.Scores[i] in [0, 1] belongs to documents[i]; sort descending to rank.
/// </code>
/// </example>
public interface ITextReranker : ITaskClient
{
    /// <summary>Longest (query, document) input in tokens: the model's limit, at most <see cref="TaskLoadOptions.MaxBatchTokens"/>.</summary>
    public int MaxSequenceLength { get; }

    /// <summary>Scores every document against the query (batched); results are in document order.</summary>
    /// <param name="query">The query.</param>
    /// <param name="documents">The documents to score.</param>
    /// <param name="options">Instruction and truncation, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    /// <returns>The scores, logits and token counts, in document order.</returns>
    public Task<RerankResult> RerankAsync(string query, IReadOnlyList<string> documents, RerankOptions? options = null, CancellationToken cancellationToken = default);
}
