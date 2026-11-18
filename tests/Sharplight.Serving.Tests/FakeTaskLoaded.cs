using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Serving.Tests;

/// <summary>A task-layer model (kind <c>Task</c>) holding the given clients.</summary>
internal sealed class FakeTaskLoaded(params ITaskClient[] clients) : LoadedModel, ITaskClientModel
{
    public override ModelKind Kind => ModelKind.Task;

    public TClient? FindClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault();

    public override void Dispose()
    {
    }
}
