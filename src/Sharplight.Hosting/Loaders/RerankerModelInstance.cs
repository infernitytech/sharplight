using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>A loaded reranker, owned until disposed (or wrapping a reranker owned elsewhere).</summary>
/// <param name="reranker">The reranker.</param>
/// <param name="residentBytes">Device bytes it holds (0 = unknown).</param>
/// <param name="ownsReranker">Dispose the reranker with the instance (when it is disposable).</param>
public sealed class RerankerModelInstance(ITextReranker reranker, long residentBytes, bool ownsReranker = true) : LoadedModel, IRerankerModel
{
    /// <inheritdoc/>
    public override ModelKind Kind => ModelKind.Reranker;

    /// <inheritdoc/>
    public override long ResidentBytes => residentBytes;

    /// <inheritdoc/>
    public ITextReranker Reranker => reranker;

    /// <inheritdoc/>
    public override void Dispose()
    {
        if (ownsReranker && reranker is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
