using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>Loads <see cref="ModelKind.Task"/> models with <see cref="TaskModelLoader"/> (the client of <see cref="HostedModelSettings.Task"/>).</summary>
public sealed class TaskModelLoaderFactory : IHostedModelLoaderFactory
{
    /// <inheritdoc/>
    public bool CanLoad(HostedModelSettings model) => model.Kind == ModelKind.Task;

    /// <inheritdoc/>
    public IHostedModelLoader CreateLoader(HostedModelLoadContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var settings = context.Settings;
        var task = settings.Task ?? throw new ArgumentException($"Task model '{settings.Name}' names no task kind ('task').", nameof(context));
        return new PlacedModelLoader(ModelKind.Task, context, () => FileFootprint.Estimate(settings.Path, settings.Auxiliary.Values), placement =>
        {
            var options = new TaskLoadOptions
            {
                ModelPath = settings.Path,
                Devices = [.. placement.Devices],
                Fallback = placement.Fallback,
                ExpertsOnDisk = settings.ExpertsOnDisk,
                ExpertCacheBytes = settings.ExpertCacheBytes,
                ContextLength = settings.ContextLength,
                MaxBatchTokens = settings.BatchTokens > 0 ? settings.BatchTokens : new TaskLoadOptions().MaxBatchTokens,
                AllowReducedPrecision = settings.ReducedPrecision,
                BlockScaledFp8 = settings.BlockScaledFp8,
                Auxiliary = new Dictionary<string, string>(settings.Auxiliary, StringComparer.Ordinal),
            };
            context.Options.ConfigureLoad?.Invoke(settings, options);
            IReadOnlyList<IMemoryDomain> pools = [.. placement.Devices.Select(slot => new ComputeDeviceMemoryDomain(slot.Device))];
            return new TaskModelLoader(task, options, pools, context.Services);
        });
    }
}
