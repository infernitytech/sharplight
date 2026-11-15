using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Finds the providers and devices the subcommands compile for: a live device (<see cref="DeviceSpec"/>), every device, or a device profile.</summary>
/// <param name="providers">The execution providers of the container.</param>
internal sealed class DeviceOpener(ExecutionProviderCatalog providers)
{
    private const string CpuProviderName = "cpu";

    /// <summary>Opens a live device.</summary>
    /// <exception cref="SampleException">The provider is unknown or unavailable, or the index is out of range.</exception>
    public IComputeDevice Open(DeviceSpec spec, DeviceOptions options)
    {
        var provider = Provider(spec.Provider);
        return provider.OpenDevice(SampleRuntime.GetDevice(provider, spec.Index), options);
    }

    /// <summary>The provider and device a <see cref="DeviceSpec"/> names.</summary>
    /// <exception cref="SampleException">The provider is unknown or unavailable, or the index is out of range.</exception>
    public (IExecutionProvider Provider, DeviceInfo Device) Find(DeviceSpec spec)
    {
        var provider = Provider(spec.Provider);
        return (provider, SampleRuntime.GetDevice(provider, spec.Index));
    }

    /// <summary>Every registered provider (available or not: the precompiler reports why it skips one).</summary>
    public IReadOnlyList<IExecutionProvider> All() => [.. providers.Discover().Select(p => p.Provider)];

    /// <summary>The registered provider whose portable stage generates code for <paramref name="profile"/>; needs no driver.</summary>
    /// <exception cref="SampleException">No registered provider compiles for the profile.</exception>
    public IExecutionProvider ForProfile(DeviceProfile profile)
    {
        if (!providers.TryFindForProfile(profile, out var provider))
        {
            throw new SampleException($"No registered provider compiles for {profile}.", ExitCodes.InvalidInput);
        }

        return provider;
    }

    /// <summary>The CPU device (fallback for ops an accelerator cannot run).</summary>
    public IComputeDevice OpenCpu()
    {
        var provider = Provider(CpuProviderName);
        return provider.OpenDevice(SampleRuntime.GetDevice(provider, 0));
    }

    private IExecutionProvider Provider(string name)
    {
        var discovered = providers.Discover();
        var match = discovered.FirstOrDefault(p => p.Provider.Name == name);
        if (match is null)
        {
            var known = string.Join(", ", discovered.Select(p => p.Provider.Name));
            throw new SampleException($"Unknown provider '{name}'; available: {known}.", ExitCodes.InvalidInput);
        }

        if (!match.Availability.IsAvailable)
        {
            throw new SampleException($"Provider '{name}' is not available: {match.Availability.Reason}");
        }

        return match.Provider;
    }
}
