using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Cli;

/// <summary>Opens the requested devices, loads the model onto them and creates the decode session.</summary>
internal static class ModelLoader
{
    public static LoadedModel Load(CliSettings settings, string modelPath)
    {
        var runtime = settings.Runtime;
        var provider = SampleRuntime.SelectProvider(runtime.UseCpu);
        var deviceInfo = SampleRuntime.GetDevice(provider, runtime.DeviceIndex);

        var deviceOptions = new DeviceOptions();
        foreach (var feature in runtime.DisabledFeatures)
        {
            deviceOptions.DisabledFeatures.Add(feature);
        }

        var slots = settings.SplitWeights
            .Select(weight => new DeviceSlot
            {
                Device = provider.OpenDevice(deviceInfo, deviceOptions),
                Weight = weight,
                MemoryLimit = settings.GpuMemoryLimitBytes,
            })
            .ToList();
        var fallback = runtime.CpuOffload ? OpenCpuFallback() : null;

        var loadOptions = new LlmLoadOptions
        {
            Auxiliary = settings.MmprojPath is { } mmproj ? [("mmproj", mmproj)] : [],
            ImageTokens = settings.ImageTokens,
            Devices = slots,
            Fallback = fallback,
            ContextLength = runtime.ContextLength,
            MaxBatchTokens = runtime.BatchTokens,
            Log = Console.Error.WriteLine,
        };

        var model = LlmModel.Load(modelPath, loadOptions);
        var graph = settings.ImagePaths.Count > 0 ? WellKnown.DecodeMultimodalGraph : WellKnown.DecodeGraph;
        return new LoadedModel(model, model.CreateSession(graph), slots, fallback);
    }

    private static DeviceSlot OpenCpuFallback()
    {
        var cpu = new CpuExecutionProvider();
        return new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
    }
}
