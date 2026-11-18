using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;

namespace Sharplight.Hosting.Loaders;

/// <summary>
/// Loads an embedding or reranking model onto devices shared with the other models of the host, for applications that place models
/// themselves (the <see cref="IHostedModelLoaderFactory"/> route places them from <see cref="HostedModelSettings"/>). The footprint is
/// estimated from the files before anything is read, like <see cref="TaskModelLoader"/>.
/// </summary>
/// <param name="kind"><see cref="ModelKind.Embedding"/> or <see cref="ModelKind.Reranker"/>.</param>
/// <param name="options">The load options (<see cref="EmbeddingLoadOptions"/> for embedders, <see cref="RerankerLoadOptions"/> for rerankers; path and devices set).</param>
/// <param name="devices">The devices of the options, as memory pools of the host.</param>
/// <param name="services">The services that provide the <see cref="ModelLoader"/>.</param>
public sealed class RetrievalModelLoader(ModelKind kind, PackedModelLoadOptions options, IReadOnlyList<IMemoryDomain> devices, IServiceProvider services)
    : IHostedModelLoader
{
    /// <inheritdoc/>
    public ModelKind Kind => kind;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        long bytes = FileFootprint.Estimate(options.ModelPath ?? "", options.Auxiliary.Values);
        return new FileSizedPreparedModel(FileSizedPreparedModel.Spread(bytes, devices), () => Load(bytes));
    }

    private LoadedModel Load(long bytes)
    {
        var loader = services.GetRequiredService<ModelLoader>();
        string path = options.ModelPath ?? throw new InvalidOperationException("The load options have no model path.");
        return kind switch
        {
            ModelKind.Embedding => new EmbeddingModelInstance(EmbeddingModel.Load(loader, path, options as EmbeddingLoadOptions ?? throw new ArgumentException("Embedding models load with EmbeddingLoadOptions.")), bytes),
            ModelKind.Reranker => new RerankerModelInstance(Reranker.Load(loader, path, options as RerankerLoadOptions ?? throw new ArgumentException("Rerankers load with RerankerLoadOptions.")), bytes),
            _ => throw new NotSupportedException($"{kind} is not a retrieval model kind."),
        };
    }
}
