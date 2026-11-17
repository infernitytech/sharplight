using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;

namespace Sharplight.Hosting.Loaders;

/// <summary>Loads <see cref="ModelKind.Reranker"/> models (<see cref="Reranker"/>); sized from the files.</summary>
public sealed class RerankerModelLoaderFactory : IHostedModelLoaderFactory
{
    /// <inheritdoc/>
    public bool CanLoad(HostedModelSettings model) => model.Kind == ModelKind.Reranker;

    /// <inheritdoc/>
    public IHostedModelLoader CreateLoader(HostedModelLoadContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var settings = context.Settings;
        var loader = context.Services.GetRequiredService<ModelLoader>();
        return PackedLoaders.Create(ModelKind.Reranker, context, () => new RerankerLoadOptions(), (options, bytes) =>
            new RerankerModelInstance(Reranker.Load(loader, settings.Path, options), bytes));
    }
}
