using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>A loaded embedding model, owned until disposed (or wrapping an embedder owned elsewhere).</summary>
/// <param name="embedder">The embedder.</param>
/// <param name="residentBytes">Device bytes it holds (0 = unknown).</param>
/// <param name="ownsEmbedder">Dispose the embedder with the instance (when it is disposable).</param>
public sealed class EmbeddingModelInstance(ITextEmbedder embedder, long residentBytes, bool ownsEmbedder = true) : LoadedModel, IEmbeddingModel
{
    /// <inheritdoc/>
    public override ModelKind Kind => ModelKind.Embedding;

    /// <inheritdoc/>
    public override long ResidentBytes => residentBytes;

    /// <inheritdoc/>
    public ITextEmbedder Embedder => embedder;

    /// <inheritdoc/>
    public override void Dispose()
    {
        if (ownsEmbedder && embedder is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
