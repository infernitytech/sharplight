using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>A task-layer model that finished loading: the model and its clients, owned until disposed.</summary>
/// <param name="model">The loaded model.</param>
/// <param name="clients">The clients created for it.</param>
/// <param name="residentBytes">Device bytes it holds (0 = unknown).</param>
/// <param name="kind">The host kind it is registered as (<see cref="TaskKindClients.HostedKind"/>).</param>
public sealed class TaskModelInstance(TaskModel model, IReadOnlyList<ITaskClient> clients, long residentBytes, ModelKind kind = ModelKind.Task)
    : LoadedModel, ITaskClientModel
{
    /// <inheritdoc/>
    public override ModelKind Kind => kind;

    /// <inheritdoc/>
    public override long ResidentBytes => residentBytes;

    /// <summary>The loaded model.</summary>
    public TaskModel Model => model;

    /// <inheritdoc/>
    public override Runtime.Experts.ExpertStore? DiskExperts => model.DiskExperts is { IsActive: true } store ? store : null;

    /// <summary>The client of type <typeparamref name="TClient"/>.</summary>
    /// <exception cref="NotSupportedException">The model has no such client.</exception>
    public TClient GetClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault() ?? throw new NotSupportedException($"The model has no {typeof(TClient).Name}.");

    /// <inheritdoc/>
    public TClient? FindClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault();

    /// <inheritdoc/>
    public override void Dispose()
    {
        foreach (var client in clients)
        {
            client.Dispose();
        }

        model.Dispose();
    }
}
