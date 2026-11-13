using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Opens the devices the subcommands compile on: a live device (<see cref="DeviceSpec"/>) or a device profile.</summary>
internal static class DeviceOpener
{
    private const string CpuProviderName = "cpu";

    /// <summary>Opens a live device.</summary>
    /// <exception cref="SampleException">The provider is unknown or unavailable, or the index is out of range.</exception>
    public static IComputeDevice Open(DeviceSpec spec, DeviceOptions options)
    {
        var provider = Provider(spec.Provider);
        return provider.OpenDevice(SampleRuntime.GetDevice(provider, spec.Index), options);
    }

    /// <summary>Opens a compile-only device for <paramref name="profile"/> through its provider; needs no driver.</summary>
    /// <exception cref="SampleException">No registered provider compiles for the profile.</exception>
    public static IComputeDevice OpenProfile(DeviceProfile profile, DeviceOptions options)
    {
        try
        {
            return ProviderRegistry.OpenProfile(profile, options);
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, ExitCodes.InvalidInput);
        }
    }

    /// <summary>The CPU device (fallback for ops an accelerator cannot run).</summary>
    public static IComputeDevice OpenCpu()
    {
        var provider = Provider(CpuProviderName);
        return provider.OpenDevice(SampleRuntime.GetDevice(provider, 0));
    }

    private static IExecutionProvider Provider(string name)
    {
        var providers = ProviderRegistry.Discover();
        var match = providers.FirstOrDefault(p => p.Provider.Name == name);
        if (match.Provider is null)
        {
            var known = string.Join(", ", providers.Select(p => p.Provider.Name));
            throw new SampleException($"Unknown provider '{name}'; available: {known}.", ExitCodes.InvalidInput);
        }

        if (!match.Availability.IsAvailable)
        {
            throw new SampleException($"Provider '{name}' is not available: {match.Availability.Reason}");
        }

        return match.Provider;
    }
}
