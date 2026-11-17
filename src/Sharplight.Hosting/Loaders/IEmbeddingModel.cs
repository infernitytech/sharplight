using Sharplight.Llm.Embeddings;

namespace Sharplight.Hosting.Loaders;

/// <summary>A loaded model of kind <c>Embedding</c>: what a lease on it gives access to.</summary>
public interface IEmbeddingModel
{
    /// <summary>The embedder; valid while the lease is held.</summary>
    public ITextEmbedder Embedder { get; }
}
