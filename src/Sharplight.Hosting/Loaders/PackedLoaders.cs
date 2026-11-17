using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>The shared loader of the packed-sequence models (embedders, rerankers): placed per load, sized from the files.</summary>
internal static class PackedLoaders
{
    public static IHostedModelLoader Create<TOptions>(ModelKind kind, HostedModelLoadContext context, Func<TOptions> createOptions, Func<TOptions, long, LoadedModel> load)
        where TOptions : PackedModelLoadOptions
    {
        var settings = context.Settings;
        long Estimate() => FileFootprint.Estimate(settings.Path, settings.Auxiliary.Values);
        return new PlacedModelLoader(kind, context, Estimate, placement => new Placed<TOptions>(kind, context, createOptions(), placement, Estimate(), load));
    }

    private sealed class Placed<TOptions>(ModelKind kind, HostedModelLoadContext context, TOptions options, DevicePlacement placement, long bytes,
        Func<TOptions, long, LoadedModel> load) : IHostedModelLoader
        where TOptions : PackedModelLoadOptions
    {
        public ModelKind Kind => kind;

        public IPreparedModel Prepare(CancellationToken ct)
        {
            var settings = context.Settings;
            options.ModelPath = settings.Path;
            options.Devices = [.. placement.Devices];
            options.Fallback = placement.Fallback;
            options.AllowReducedPrecision = settings.ReducedPrecision;
            if (settings.BatchTokens > 0)
            {
                options.MaxBatchTokens = settings.BatchTokens;
            }

            foreach (var (role, path) in settings.Auxiliary)
            {
                options.Auxiliary[role] = path;
            }

            context.Options.ConfigureLoad?.Invoke(settings, options);
            IReadOnlyList<IMemoryDomain> pools = [.. placement.Devices.Select(slot => new ComputeDeviceMemoryDomain(slot.Device))];
            return new FileSizedPreparedModel(FileSizedPreparedModel.Spread(bytes, pools), () => load(options, bytes));
        }
    }
}
