using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>A task-layer model that finished loading: the model and its clients, owned until disposed.</summary>
public sealed class TaskModelInstance(TaskModel model, IReadOnlyList<ITaskClient> clients, long residentBytes) : LoadedModel
{
    /// <inheritdoc/>
    public override ModelKind Kind => ModelKind.Task;

    /// <inheritdoc/>
    public override long ResidentBytes => residentBytes;

    /// <summary>The loaded model.</summary>
    public TaskModel Model => model;

    /// <summary>The client of type <typeparamref name="TClient"/>.</summary>
    /// <exception cref="NotSupportedException">The model has no such client.</exception>
    public TClient GetClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault() ?? throw new NotSupportedException($"The model has no {typeof(TClient).Name}.");

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
