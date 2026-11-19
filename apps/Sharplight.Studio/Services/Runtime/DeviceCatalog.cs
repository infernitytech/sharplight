using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Lists the devices of the available execution providers, GPUs first and otherwise in the catalog's preference order (on Apple silicon
/// the Metal device ahead of the same GPU through Vulkan / MoltenVK, on NVIDIA GPUs Vulkan ahead of CUDA), so the first entry is the default
/// device. A GPU that several providers list (the same UUID) gets one entry per provider, marked with the provider's title.
/// </summary>
internal static class DeviceCatalog
{
    /// <summary>The oldest CUDA driver that compiles the PTX of the shipped NVRTC (CUDA 13.0, the R580 branch).</summary>
    public const int MinimumCudaDriverMajor = 13;

    /// <summary>Every device of every available provider; providers that fail to enumerate are skipped.</summary>
    /// <param name="providers">The execution providers.</param>
    public static DeviceDiscovery Discover(ExecutionProviderCatalog providers)
    {
        var found = new List<(ComputeDeviceOption Option, IExecutionProvider Provider, DeviceInfo Info)>();
        var unavailable = new Dictionary<string, string>(StringComparer.Ordinal);
        var unusable = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (provider, availability) in providers.Discover())
        {
            if (!availability.IsAvailable)
            {
                unavailable.TryAdd(provider.Name, availability.Reason);
                continue;
            }

            IReadOnlyList<DeviceInfo> devices;
            try
            {
                devices = provider.GetDevices();
            }
            catch (Exception error)
            {
                unavailable.TryAdd(provider.Name, error.Message);
                continue;
            }

            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var info in devices)
            {
                bool gpu = info.Kind is not DeviceKind.Cpu;
                string? key = gpu ? GpuKey(info, counts) : null;
                var option = new ComputeDeviceOption($"{info.Provider}:{info.Index}", info.Name, provider.Name, gpu, info.TotalMemory,
                    [.. info.Features.Order(StringComparer.Ordinal)], key);
                if (CudaDriverProblem(info) is { } problem)
                {
                    unusable[option.Id] = problem;
                }

                found.Add((option, provider, info));
            }
        }

        // A GPU listed by more than one provider is told apart by the provider's title.
        var shared = found.Where(entry => entry.Option.GpuKey is not null).GroupBy(entry => entry.Option.GpuKey).Where(group => group.Count() > 1)
            .Select(group => group.Key).ToHashSet(StringComparer.Ordinal);
        for (int i = 0; i < found.Count; i++)
        {
            var entry = found[i];
            if (entry.Option.GpuKey is { } key && shared.Contains(key))
            {
                found[i] = (entry.Option with { Variant = ProviderTitles.Of(entry.Option.Provider) }, entry.Provider, entry.Info);
            }
        }

        // OrderBy is stable: within GPUs and CPUs the providers keep their registration (preference) order.
        return new DeviceDiscovery([.. found.OrderBy(entry => entry.Option.IsGpu ? 0 : 1)], unavailable, unusable);
    }

    /// <summary>
    /// The identity of a GPU across providers: its UUID (Vulkan reports it as hex, CUDA as a dashed GUID: both normalized), else the vendor,
    /// name and the number of GPUs of that name seen so far in the same provider.
    /// </summary>
    internal static string GpuKey(DeviceInfo info, Dictionary<string, int> counts)
    {
        if (!string.IsNullOrWhiteSpace(info.Uuid))
        {
            return info.Uuid.Replace("-", "", StringComparison.Ordinal).ToUpperInvariant();
        }

        string name = $"{info.Vendor}:{info.Name}";
        counts[name] = counts.TryGetValue(name, out int seen) ? seen + 1 : 0;
        return string.Create(CultureInfo.InvariantCulture, $"{name}#{counts[name]}");
    }

    /// <summary>Why a CUDA device cannot be used (its driver is older than the PTX the shipped compiler emits), or null.</summary>
    internal static string? CudaDriverProblem(DeviceInfo info)
    {
        if (info.Provider != ProviderTitles.Cuda || !info.Capabilities.TryGetValue("cuda_driver", out var text))
        {
            return null;
        }

        if (int.TryParse(text.Split('.')[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int major) && major < MinimumCudaDriverMajor)
        {
            return $"the NVIDIA driver supports CUDA {text}; CUDA {MinimumCudaDriverMajor}.0 (driver R580 or newer) is needed. Update the NVIDIA driver";
        }

        return null;
    }
}
