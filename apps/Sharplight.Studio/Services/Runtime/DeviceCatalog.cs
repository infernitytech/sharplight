using Sharplight.Runtime;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Lists the devices of the available execution providers, GPUs first.</summary>
internal static class DeviceCatalog
{
    /// <summary>Every device of every available provider; providers that fail to enumerate are skipped.</summary>
    public static IReadOnlyList<(ComputeDeviceOption Option, IExecutionProvider Provider, DeviceInfo Info)> Discover()
    {
        var found = new List<(ComputeDeviceOption, IExecutionProvider, DeviceInfo)>();
        foreach (var (provider, availability) in ProviderRegistry.Discover())
        {
            if (!availability.IsAvailable)
            {
                continue;
            }

            IReadOnlyList<DeviceInfo> devices;
            try
            {
                devices = provider.GetDevices();
            }
            catch (Exception)
            {
                continue;
            }

            foreach (var info in devices)
            {
                bool gpu = info.Kind is not DeviceKind.Cpu;
                var option = new ComputeDeviceOption($"{info.Provider}:{info.Index}", info.Name, provider.Name, gpu, info.TotalMemory,
                    [.. info.Features.Order(StringComparer.Ordinal)]);
                found.Add((option, provider, info));
            }
        }

        return [.. found.OrderBy(entry => entry.Item1.IsGpu ? 0 : 1).ThenBy(entry => entry.Item1.Id, StringComparer.Ordinal)];
    }
}
