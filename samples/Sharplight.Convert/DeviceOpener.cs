using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Opens the live devices the subcommands compile on (<see cref="DeviceSpec"/>).</summary>
/// <param name="providers">The execution providers to open devices on.</param>
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
