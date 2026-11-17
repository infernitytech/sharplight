using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;

namespace Sharplight.Hosting.Loaders;

/// <summary>Loads <see cref="ModelKind.Embedding"/> models (<see cref="EmbeddingModel"/>); sized from the files.</summary>
public sealed class EmbeddingModelLoaderFactory : IHostedModelLoaderFactory
{
    /// <inheritdoc/>
    public bool CanLoad(HostedModelSettings model) => model.Kind == ModelKind.Embedding;

    /// <inheritdoc/>
    public IHostedModelLoader CreateLoader(HostedModelLoadContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var settings = context.Settings;
        var loader = context.Services.GetRequiredService<ModelLoader>();
        return PackedLoaders.Create(ModelKind.Embedding, context, () => new EmbeddingLoadOptions(), (options, bytes) =>
            new EmbeddingModelInstance(EmbeddingModel.Load(loader, settings.Path, options), bytes));
    }
}
