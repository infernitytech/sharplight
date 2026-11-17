using Sharplight.Llm.Embeddings;

namespace Sharplight.Hosting.Loaders;

/// <summary>A loaded model of kind <c>Reranker</c>: what a lease on it gives access to.</summary>
public interface IRerankerModel
{
    /// <summary>The reranker; valid while the lease is held.</summary>
    public ITextReranker Reranker { get; }
}
