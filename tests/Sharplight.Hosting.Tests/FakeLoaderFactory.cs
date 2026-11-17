using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Tests;

/// <summary>Loads every model as a <see cref="FakeLoadedModel"/> of the size its path names (<c>fake://2G</c>) on one pool.</summary>
public sealed class FakeLoaderFactory(FakePool pool) : IHostedModelLoaderFactory
{
    private readonly FakePool _pool = pool;

    public List<string> Loaded { get; } = [];

    public bool CanLoad(HostedModelSettings model) => model.Path.StartsWith("fake://", StringComparison.Ordinal);

    public IHostedModelLoader CreateLoader(HostedModelLoadContext context) => new Loader(this, context.Settings);

    private sealed class Loader(FakeLoaderFactory owner, HostedModelSettings settings) : IHostedModelLoader, IPreparedModel
    {
        private long Bytes => MemorySize.Parse(settings.Path["fake://".Length..]);

        public ModelKind Kind => settings.Kind;

        public IReadOnlyList<MemoryRequirement> Requirements => [new MemoryRequirement(owner._pool, Bytes)];

        public bool CanOffload => false;

        public IPreparedModel Prepare(CancellationToken ct) => this;

        public LoadedModel Load(long? primaryLimit)
        {
            lock (owner.Loaded)
            {
                owner.Loaded.Add(settings.Name);
            }

            return new FakeLoadedModel(owner._pool, Bytes, settings.Kind);
        }

        public void Dispose()
        {
            // The loader is reused for every load.
        }
    }
}
