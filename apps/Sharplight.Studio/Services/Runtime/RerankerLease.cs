using Sharplight.Llm.Embeddings;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A claim on the reranker of a resident reranking model (<see cref="ILoadedModel.AcquireRerankerAsync"/>): while it is open the model
/// is neither evicted nor unloaded. Dispose when the run is done.
/// </summary>
internal sealed class RerankerLease : IDisposable
{
    private readonly IDisposable? _lease;

    /// <param name="reranker">The reranker, valid until the lease is disposed.</param>
    /// <param name="lease">What to release with the lease (the host's model lease), or null.</param>
    public RerankerLease(ITextReranker reranker, IDisposable? lease = null)
    {
        Reranker = reranker;
        _lease = lease;
    }

    /// <summary>The reranker.</summary>
    public ITextReranker Reranker { get; }

    /// <inheritdoc/>
    public void Dispose() => _lease?.Dispose();
}
