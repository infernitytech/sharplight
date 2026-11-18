using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A task-layer model for a <see cref="ModelHost"/> holding the given clients, without device memory.</summary>
internal sealed class HostedTaskModel(params ITaskClient[] clients) : LoadedModel, ITaskClientModel
{
    public override ModelKind Kind => ModelKind.Task;

    public TClient? FindClient<TClient>()
        where TClient : class, ITaskClient
        => clients.OfType<TClient>().FirstOrDefault();

    public override void Dispose()
    {
    }

    /// <summary>Registers a model on <paramref name="host"/> that loads <paramref name="create"/> on first use.</summary>
    public static void Register(ModelHost host, string name, ModelKind kind, Func<LoadedModel> create)
        => host.Register(new ModelRegistration { Name = name, Loader = new Loader(kind, create) });

    private sealed class Loader(ModelKind kind, Func<LoadedModel> create) : IHostedModelLoader
    {
        public ModelKind Kind => kind;

        public IPreparedModel Prepare(CancellationToken ct) => new Prepared(create);
    }

    private sealed class Prepared(Func<LoadedModel> create) : IPreparedModel
    {
        public IReadOnlyList<MemoryRequirement> Requirements => [];

        public bool CanOffload => false;

        public LoadedModel Load(long? primaryLimit) => create();

        public void Dispose()
        {
        }
    }
}
