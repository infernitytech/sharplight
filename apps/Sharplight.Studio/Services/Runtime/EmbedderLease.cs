using Sharplight.Llm.Embeddings;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A claim on the embedder of a resident embedding model (<see cref="ILoadedModel.AcquireEmbedderAsync"/>): while it is open the model
/// is neither evicted nor unloaded. Dispose when the run is done.
/// </summary>
internal sealed class EmbedderLease : IDisposable
{
    private readonly IDisposable? _lease;

    /// <param name="embedder">The embedder, valid until the lease is disposed.</param>
    /// <param name="lease">What to release with the lease (the host's model lease), or null.</param>
    public EmbedderLease(ITextEmbedder embedder, IDisposable? lease = null)
    {
        Embedder = embedder;
        _lease = lease;
    }

    /// <summary>The embedder.</summary>
    public ITextEmbedder Embedder { get; }

    /// <inheritdoc/>
    public void Dispose() => _lease?.Dispose();
}
