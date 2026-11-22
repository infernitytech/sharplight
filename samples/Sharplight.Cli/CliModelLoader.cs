using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Vision;

namespace Sharplight.Cli;

/// <summary>Opens the requested devices, loads the model onto them and creates the decode session.</summary>
internal static class CliModelLoader
{
    public static LoadedModel Load(SampleRuntime sampleRuntime, CliSettings settings, string modelPath, CancellationToken cancellationToken)
    {
        var runtime = settings.Runtime;
        var provider = sampleRuntime.SelectProvider(runtime);
        var deviceInfo = SampleRuntime.GetDevice(provider, runtime.DeviceIndex);

        var deviceOptions = new DeviceOptions();
        foreach (var feature in runtime.DisabledFeatures)
        {
            deviceOptions.DisabledFeatures.Add(feature);
        }

        var slots = settings.GpuCount > 1
            ? OpenGpus(provider, runtime.DeviceIndex, settings, deviceOptions)
            : settings.SplitWeights
                .Select(weight => new DeviceSlot
                {
                    Device = provider.OpenDevice(deviceInfo, deviceOptions),
                    Weight = weight,
                    MemoryLimit = settings.GpuMemoryLimitBytes,
                })
                .ToList();
        var fallback = runtime.CpuOffload ? OpenCpuFallback() : null;
        var progress = new ConsoleLoadProgress();

        var loadOptions = new LlmLoadOptions
        {
            Auxiliary = Auxiliary(settings.MmprojPath, settings.DraftPath),
            ImageTokens = settings.ImageTokens,
            Devices = slots,
            Fallback = fallback,
            ExpertsOnDisk = runtime.ExpertsOnDisk,
            ExpertCacheBytes = runtime.ExpertCacheBytes,
            ContextLength = runtime.ContextLength,
            MaxBatchTokens = runtime.BatchTokens,
            KvCacheType = runtime.KvCacheType,
            AllowReducedPrecision = runtime.ReducedPrecision,
            Log = progress.Log,
            Progress = progress,
            CancellationToken = cancellationToken,
        };

        var model = LlmModel.Load(sampleRuntime.Loader, modelPath, loadOptions);
        var graph = settings.ImagePaths.Count > 0 ? WellKnown.DecodeMultimodalGraph : WellKnown.DecodeGraph;
        return new LoadedModel(model, graph, slots, fallback);
    }

    /// <summary>
    /// One slot per physical GPU from <paramref name="first"/> on; the layer shares are the <c>--split</c> weights when there is one per GPU,
    /// else every GPU's available memory.
    /// </summary>
    private static List<DeviceSlot> OpenGpus(IExecutionProvider provider, int first, CliSettings settings, DeviceOptions deviceOptions)
    {
        bool weighted = settings.SplitWeights.Count == settings.GpuCount;
        return [.. Enumerable.Range(0, settings.GpuCount).Select(i => new DeviceSlot
        {
            Device = provider.OpenDevice(SampleRuntime.GetDevice(provider, first + i), deviceOptions),
            Weight = weighted ? settings.SplitWeights[i] : null,
            MemoryLimit = settings.GpuMemoryLimitBytes,
        })];
    }

    private static Dictionary<string, string> Auxiliary(string? mmproj, string? draft)
    {
        var auxiliary = new Dictionary<string, string>(StringComparer.Ordinal);
        if (mmproj is not null)
        {
            auxiliary[AuxiliaryRoles.Mmproj] = mmproj;
        }

        if (draft is not null)
        {
            auxiliary[AuxiliaryRoles.Draft] = draft;
        }

        return auxiliary;
    }

    private static DeviceSlot OpenCpuFallback()
    {
        var cpu = new CpuExecutionProvider();
        return new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
    }
}
