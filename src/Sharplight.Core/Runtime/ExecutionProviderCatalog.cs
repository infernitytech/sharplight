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
    /// Opens a compile-only device for <paramref name="profile"/> (see <see cref="IOfflineCompilationProvider.OpenProfile"/>) through
    /// the first provider that compiles for it, without probing drivers.
    /// </summary>
    /// <param name="profile">The device profile.</param>
    /// <param name="options">Device options, or <see langword="null"/> for the defaults.</param>
    /// <returns>The device; the caller disposes it.</returns>
    /// <exception cref="NotSupportedException">No registered provider compiles for the profile.</exception>
    public IComputeDevice OpenProfile(DeviceProfile profile, DeviceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        foreach (var provider in _providers)
        {
            if (provider is IOfflineCompilationProvider offline && offline.CanCompileFor(profile))
            {
                return offline.OpenProfile(profile, options);
            }
        }

        throw new NotSupportedException($"No registered provider compiles for {profile}.");
    }
}
