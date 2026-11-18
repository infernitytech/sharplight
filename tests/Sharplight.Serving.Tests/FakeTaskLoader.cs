using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

/// <summary>Loads a <see cref="FakeTaskLoaded"/> (or any instance) of a given kind without device memory.</summary>
internal sealed class FakeTaskLoader(ModelKind kind, Func<LoadedModel> create) : IHostedModelLoader
{
    public ModelKind Kind => kind;

    public int Loads { get; private set; }

    public IPreparedModel Prepare(CancellationToken ct) => new Prepared(this);

    private sealed class Prepared(FakeTaskLoader owner) : IPreparedModel
    {
        public IReadOnlyList<MemoryRequirement> Requirements => [];

        public bool CanOffload => false;

        public LoadedModel Load(long? limit)
        {
            owner.Loads++;
            return owner._create();
        }

        public void Dispose()
        {
        }
    }

    private readonly Func<LoadedModel> _create = create;
}
