using Sharplight.Runtime;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>What <see cref="DeviceCatalog.Discover"/> found: the devices of the available providers and what the others reported.</summary>
/// <param name="Entries">Every device of every available provider: the option, its provider and its info.</param>
/// <param name="Unavailable">Providers that are registered but cannot run here, by provider name, with the reason.</param>
/// <param name="Unusable">Devices that are listed but cannot be used, by option id, with the reason (a CUDA driver older than the shipped compiler's PTX).</param>
internal sealed record DeviceDiscovery(IReadOnlyList<(ComputeDeviceOption Option, IExecutionProvider Provider, DeviceInfo Info)> Entries,
    IReadOnlyDictionary<string, string> Unavailable, IReadOnlyDictionary<string, string> Unusable)
{
    /// <summary>The options of <see cref="Entries"/>.</summary>
    public IReadOnlyList<ComputeDeviceOption> Options => [.. Entries.Select(entry => entry.Option)];
}
