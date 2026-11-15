using System.Collections.Concurrent;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Precompile;

/// <summary>
/// <see cref="IPrecompileService"/> over <see cref="PackagePrecompiler"/> and every registered execution provider (unavailable ones are
/// reported as skipped), compiling with the default load settings so the stored code is what a load from the library asks for.
/// </summary>
internal sealed class PrecompileService(ISettingsService settings, EngineServices engine) : IPrecompileService
{
    private readonly ConcurrentDictionary<string, bool> _busy = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public bool IsBusy(string path) => _busy.ContainsKey(Path.GetFullPath(path));

    /// <inheritdoc/>
    public async Task<PackagePrecompileReport> PrecompileAsync(string path, IProgress<PrecompileProgress> progress, CancellationToken cancellationToken)
    {
        var key = Path.GetFullPath(path);
        if (!_busy.TryAdd(key, true))
        {
            throw new InvalidOperationException($"{path} is already being precompiled.");
        }

        try
        {
            return await Task.Run(async () =>
            {
                var providers = engine.Providers.Discover().Select(p => p.Provider).ToList();
                using var cpu = OpenCpu();
                var options = CreateOptions(settings.Current, cpu, log: null, kernels: null);
                return await PackagePrecompiler.PrecompileAsync(path, providers, options, progress, cancellationToken).ConfigureAwait(false);
            }, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _busy.TryRemove(key, out _);
        }
    }

    /// <summary>
    /// The options library models are precompiled with (after a download too): the default load settings (context, batch, slots), the
    /// disabled device features, and the per-user compilation cache when it is on (faster; the stored sets still hold only the model's code).
    /// </summary>
    /// <param name="settings">The app settings.</param>
    /// <param name="fallback">Device for ops an accelerator cannot run in non-LLM graphs (the CPU), or <see langword="null"/>.</param>
    /// <param name="log">Receives progress messages, or <see langword="null"/>.</param>
    /// <param name="kernels">Receives kernel compilation progress, or <see langword="null"/>.</param>
    public static PrecompileOptions CreateOptions(AppSettings settings, IComputeDevice? fallback, Action<string>? log, IProgress<LoadProgress>? kernels) => new()
    {
        ContextLength = settings.DefaultLoad.ContextLength,
        BatchTokens = settings.DefaultLoad.BatchTokens,
        Slots = settings.DefaultLoad.Slots,
        DisabledFeatures = settings.DisabledFeatures,
        FallbackDevice = fallback,
        UseUserKernelCache = settings.UseCompilationCache,
        Log = log,
        Progress = kernels,
    };

    private IComputeDevice? OpenCpu()
    {
        return engine.Providers.TryFind("cpu", out var provider) ? provider.OpenDevice(provider.GetDevices()[0]) : null;
    }
}
