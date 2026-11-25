using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>
/// Task clients created elsewhere, registered with a host as one loaded model (e.g. an <c>ITextEmbedder</c> an application loaded
/// itself, or a test double); <see cref="TaskModelInstance"/> is the form the task loaders produce.
/// </summary>
/// <param name="clients">The clients.</param>
/// <param name="residentBytes">Device bytes they hold (0 = unknown).</param>
/// <param name="kind">The host kind (<see cref="ModelKind.Embedding"/> for an embedder, <see cref="ModelKind.Reranker"/> for a reranker, ...).</param>
/// <param name="ownsClients">Dispose the clients with the instance.</param>
public sealed class TaskClientInstance(IReadOnlyList<ITaskClient> clients, long residentBytes, ModelKind kind, bool ownsClients = true) : LoadedModel, ITaskClientModel
{
    /// <inheritdoc/>
    public override ModelKind Kind => kind;

    /// <inheritdoc/>
    public override long ResidentBytes => residentBytes;

    /// <inheritdoc/>
    public TClient? FindClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault();

    /// <inheritdoc/>
    public override void Dispose()
    {
        if (ownsClients)
        {
            foreach (var client in clients)
            {
                client.Dispose();
            }
        }
    }
}
