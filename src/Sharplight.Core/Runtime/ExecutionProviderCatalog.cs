using System.Diagnostics.CodeAnalysis;

namespace Sharplight.Runtime;

/// <summary>
/// The execution providers a host supports: the <see cref="IExecutionProvider"/> services registered with the container
/// (<c>services.AddVulkanProvider()</c>, <c>services.AddCpuProvider()</c>, or <c>services.AddExecutionProvider&lt;T&gt;()</c>), in
/// registration order (register GPUs before the CPU so that <see cref="Discover"/> lists them first). The providers are owned by the
/// container. Registered as a singleton by every provider registration; thread-safe.
/// </summary>
public sealed class ExecutionProviderCatalog
{
    private readonly IExecutionProvider[] _providers;

    /// <summary>Creates the catalog over a set of providers; the caller keeps ownership of them.</summary>
    /// <param name="providers">The providers in preference order.</param>
    public ExecutionProviderCatalog(IEnumerable<IExecutionProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        _providers = [.. providers];
    }

    /// <summary>The registered providers, available or not, in preference order.</summary>
    public IReadOnlyList<IExecutionProvider> Providers => _providers;

    /// <summary>Checks every provider's availability (a provider whose check throws is reported unavailable with the error).</summary>
    /// <returns>The providers in preference order with their availability.</returns>
    public IReadOnlyList<DiscoveredProvider> Discover()
    {
        var result = new List<DiscoveredProvider>(_providers.Length);
        foreach (var provider in _providers)
        {
            ProviderAvailability availability;
            try
            {
                availability = provider.CheckAvailability();
            }
            catch (Exception e)
            {
                availability = ProviderAvailability.Unavailable(e.Message);
            }

            result.Add(new DiscoveredProvider(provider, availability));
        }

        return result;
    }

    /// <summary>Finds the first available provider with the given name.</summary>
    /// <param name="name">The provider name (<see cref="IExecutionProvider.Name"/>, e.g. <c>vulkan</c>).</param>
    /// <param name="provider">The provider, or <see langword="null"/> when none with this name is available.</param>
    /// <returns><see langword="true"/> when an available provider was found.</returns>
    public bool TryFind(string name, [NotNullWhen(true)] out IExecutionProvider? provider)
    {
        ArgumentNullException.ThrowIfNull(name);
        provider = Discover().FirstOrDefault(d => d.Availability.IsAvailable && d.Provider.Name == name)?.Provider;
        return provider is not null;
    }

    /// <summary>
    /// Finds the first provider whose portable stage generates code for <paramref name="profile"/>
    /// (<see cref="IExecutionProvider.CanCompileFor"/>), for offline precompilation (<see cref="IExecutionProvider.CompilePortableAsync"/>),
    /// without probing drivers.
    /// </summary>
    /// <param name="profile">The device profile.</param>
    /// <param name="provider">The provider (owned by the catalog's container), or <see langword="null"/> when none compiles for the profile.</param>
    /// <returns><see langword="true"/> when a provider was found.</returns>
    public bool TryFindForProfile(DeviceProfile profile, [NotNullWhen(true)] out IExecutionProvider? provider)
    {
        ArgumentNullException.ThrowIfNull(profile);
        provider = _providers.FirstOrDefault(p => p.CanCompileFor(profile));
        return provider is not null;
    }
}
