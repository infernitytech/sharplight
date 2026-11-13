using Sharplight.Llm.Embeddings;

namespace Sharplight.Server;

/// <summary>The embedding and reranking models a host serves, by the name clients pass as <c>model</c>.</summary>
public sealed class RetrievalModels
{
    /// <summary>Embedding models by name.</summary>
    public Dictionary<string, ITextEmbedder> Embedders { get; } = new(StringComparer.Ordinal);
    /// <summary>Reranking models by name.</summary>
    public Dictionary<string, ITextReranker> Rerankers { get; } = new(StringComparer.Ordinal);
    /// <summary>Embedder used when a request names none (default: the first registered).</summary>
    public string? DefaultEmbedder { get; set; }
    /// <summary>Reranker used when a request names none (default: the first registered).</summary>
    public string? DefaultReranker { get; set; }
    /// <summary>
    /// Cut over-long inputs to the model's limit when the request does not say (<c>truncate</c>). Default false: like OpenAI's API,
    /// an over-long input is a 400 error. Rerankers always truncate (the document is shortened), as Cohere's API does.
    /// </summary>
    public bool TruncateByDefault { get; set; }
    /// <summary>Most inputs (texts or documents) in one request.</summary>
    public int MaxInputs { get; set; } = 2048;
    /// <summary>Largest accepted request body.</summary>
    public long MaxRequestBodyBytes { get; set; } = 32 * 1024 * 1024;
}
