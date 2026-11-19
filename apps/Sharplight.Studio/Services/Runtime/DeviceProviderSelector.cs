using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Decides which provider runs a GPU that more than one provider lists (an NVIDIA GPU: Vulkan and CUDA). The per-GPU choice
/// (<see cref="ProviderChoice"/>) orders the entries; <see cref="ProviderChoice.Automatic"/> is Vulkan until
/// <see cref="IProviderPerformanceHints"/> knows better. A provider that cannot be used (driver too old, not available on this machine) or that
/// failed is skipped, with a note that the device list shows and the log carries: the GPU then runs on the next provider, in practice Vulkan.
/// </summary>
internal sealed class DeviceProviderSelector
{
    private readonly DeviceDiscovery _discovery;
    private readonly Func<IReadOnlyDictionary<string, ProviderChoice>> _choices;
    private readonly IProviderPerformanceHints _hints;
    private readonly ILogger _logger;
    private readonly ConcurrentDictionary<string, string> _failed = new(StringComparer.Ordinal);

    /// <param name="discovery">The devices and what the providers reported.</param>
    /// <param name="choices">The current per-GPU choices (read on every call, so settings changes apply at once).</param>
    /// <param name="hints">Per-device knowledge for <see cref="ProviderChoice.Automatic"/>; null: none (Vulkan).</param>
    /// <param name="logger">Receives the fallback notes.</param>
    public DeviceProviderSelector(DeviceDiscovery discovery, Func<IReadOnlyDictionary<string, ProviderChoice>>? choices = null,
        IProviderPerformanceHints? hints = null, ILogger? logger = null)
    {
        _discovery = discovery;
        _choices = choices ?? (() => new Dictionary<string, ProviderChoice>());
        _hints = hints ?? new NoProviderHints();
        _logger = logger ?? NullLogger.Instance;
    }

    /// <summary>Every device: per GPU the entry in use first, then the other providers' entries, then the CPU.</summary>
    public IReadOnlyList<ComputeDeviceOption> Devices
    {
        get
        {
            var devices = new List<ComputeDeviceOption>();
            foreach (var gpu in Gpus)
            {
                devices.AddRange(gpu.Entries);
            }

            devices.AddRange(_discovery.Options.Where(option => !option.IsGpu));
            return devices;
        }
    }

    /// <summary>The physical GPUs, in discovery order.</summary>
    public IReadOnlyList<PhysicalGpu> Gpus
    {
        get
        {
            var choices = _choices();
            var gpus = new List<PhysicalGpu>();
            foreach (var group in _discovery.Entries.Where(entry => entry.Option.GpuKey is not null).GroupBy(entry => entry.Option.GpuKey!))
            {
                var entries = group.ToList();
                var options = entries.Select(entry => entry.Option).ToList();
                var choice = choices.GetValueOrDefault(group.Key, ProviderChoice.Automatic);
                string? preferred = choice switch
                {
                    ProviderChoice.Vulkan => ProviderTitles.Vulkan,
                    ProviderChoice.Cuda => ProviderTitles.Cuda,
                    _ => _hints.FasterProvider(options) ?? ProviderTitles.Vulkan,
                };
                var ordered = options.OrderBy(option => option.Provider == preferred ? 0 : 1).ToList();
                var effective = ordered.FirstOrDefault(IsUsable) ?? ordered[0];
                ordered.Remove(effective);
                ordered.Insert(0, effective);
                bool nvidia = entries.Any(entry => entry.Info.Vendor.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase));
                gpus.Add(new PhysicalGpu(group.Key, entries[0].Info.Name, entries.Max(entry => entry.Info.TotalMemory), ordered, choice, effective,
                    NoteFor(options, nvidia)));
            }

            return gpus;
        }
    }

    /// <summary>
    /// The notes about providers that are not used although the machine could have them, for the log at start (the same text the device list
    /// shows).
    /// </summary>
    public IReadOnlyList<string> Notes => [.. Gpus.Where(gpu => gpu.Note is not null).Select(gpu => $"{gpu.Name}: {gpu.Note}")];

    /// <summary>Whether the device can be used (it is not known to fail).</summary>
    /// <param name="option">A device entry.</param>
    public bool IsUsable(ComputeDeviceOption option) => !_discovery.Unusable.ContainsKey(option.Id) && !_failed.ContainsKey(option.Id);

    /// <summary>
    /// The device to use instead of <paramref name="option"/> when it cannot be used: the GPU's entry of another provider; otherwise
    /// <paramref name="option"/> itself.
    /// </summary>
    /// <param name="option">The device a load asked for.</param>
    public ComputeDeviceOption Usable(ComputeDeviceOption option)
    {
        if (option.GpuKey is null || IsUsable(option))
        {
            return option;
        }

        var gpu = Gpus.FirstOrDefault(candidate => candidate.Key == option.GpuKey);
        if (gpu is not null && gpu.Effective.Id != option.Id && IsUsable(gpu.Effective))
        {
            _logger.LogWarning("{Provider} cannot be used on {Device}: {Reason}. Using {Fallback} instead", ProviderTitles.Of(option.Provider), option.Name,
                Reason(option), ProviderTitles.Of(gpu.Effective.Provider));
            return gpu.Effective;
        }

        return option;
    }

    /// <summary>
    /// Records that <paramref name="option"/> failed (opening it or loading a model on it) and returns the entry of another provider of the same
    /// GPU to retry on, or null when there is none (the CPU, a GPU with one provider, an error that is not the provider's).
    /// </summary>
    /// <param name="option">The device that failed.</param>
    /// <param name="error">The failure.</param>
    public ComputeDeviceOption? Failover(ComputeDeviceOption option, Exception error)
    {
        if (option.GpuKey is null || error is OperationCanceledException or OutOfMemoryException)
        {
            return null;
        }

        var sibling = _discovery.Entries.Select(entry => entry.Option)
            .FirstOrDefault(candidate => candidate.GpuKey == option.GpuKey && candidate.Id != option.Id && IsUsable(candidate));
        if (sibling is null)
        {
            return null;
        }

        _failed[option.Id] = error.Message;
        _logger.LogWarning(error, "{Provider} failed on {Device}; falling back to {Fallback} for this GPU", ProviderTitles.Of(option.Provider), option.Name,
            ProviderTitles.Of(sibling.Provider));
        return sibling;
    }

    private string Reason(ComputeDeviceOption option)
        => _discovery.Unusable.TryGetValue(option.Id, out var reason) ? reason : _failed.GetValueOrDefault(option.Id, "it failed earlier");

    private string? NoteFor(IReadOnlyList<ComputeDeviceOption> entries, bool nvidia)
    {
        var cuda = entries.FirstOrDefault(entry => entry.Provider == ProviderTitles.Cuda);
        if (cuda is null)
        {
            return nvidia && _discovery.Unavailable.TryGetValue(ProviderTitles.Cuda, out var why)
                ? $"CUDA is not available on this machine ({why.TrimEnd('.')}). Vulkan is used."
                : null;
        }

        return IsUsable(cuda) ? null : $"CUDA cannot be used: {Reason(cuda).TrimEnd('.')}. Vulkan is used.";
    }
}
