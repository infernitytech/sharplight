using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

/// <summary>Loads whatever instance <see cref="Create"/> returns (embedding and reranker instances of fakes), charging no memory.</summary>
public sealed class FakeInstanceLoader(ModelKind kind, Func<LoadedModel> create) : IHostedModelLoader, IPreparedModel
{
    public ModelKind Kind => kind;

    public Func<LoadedModel> Create => create;

    public IReadOnlyList<MemoryRequirement> Requirements => [];

    public bool CanOffload => false;

    public IPreparedModel Prepare(CancellationToken ct) => this;

    public LoadedModel Load(long? primaryLimit) => create();

    public void Dispose()
    {
        // Reused for every load.
    }
}
