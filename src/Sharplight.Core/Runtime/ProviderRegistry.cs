namespace Sharplight.Runtime;

/// <summary>Registry of available execution providers (Vulkan, CUDA, CPU, ...).</summary>
public static class ProviderRegistry
{
    private static readonly List<Func<IExecutionProvider>> s_factories = [];
    private static readonly Lock s_lock = new();

    /// <summary>
    /// Adds a provider factory. Factories are called on every <see cref="Discover"/>; a factory that throws yields an unavailable provider.
    /// </summary>
    public static void Register(Func<IExecutionProvider> factory)
    {
        lock (s_lock)
        {
            s_factories.Add(factory);
        }
    }

    /// <summary>Instantiates all registered providers and reports their availability.</summary>
    public static IReadOnlyList<(IExecutionProvider Provider, ProviderAvailability Availability)> Discover()
    {
        List<Func<IExecutionProvider>> f;
        lock (s_lock)
        {
            f = [.. s_factories];
        }

        var result = new List<(IExecutionProvider, ProviderAvailability)>();
        foreach (var factory in f)
        {
            IExecutionProvider p;
            try
            {
                p = factory();
            }
            catch (Exception e)
            {
                result.Add((new UnavailableProvider(factory.Method.DeclaringType?.Name ?? "?", e.Message), ProviderAvailability.Unavailable(e.Message)));
                continue;
            }
            ProviderAvailability a;
            try
            {
                a = p.CheckAvailability();
            }
            catch (Exception e)
            {
                a = ProviderAvailability.Unavailable(e.Message);
            }
            result.Add((p, a));
        }
        return result;
    }

    /// <summary>
    /// The first registered provider whose portable stage generates code for <paramref name="profile"/>
    /// (<see cref="IExecutionProvider.CanCompileFor"/>), for offline precompilation (<see cref="IExecutionProvider.CompilePortableAsync"/>).
    /// Providers are instantiated without probing their drivers. The caller disposes the provider.
    /// </summary>
    /// <param name="profile">The device profile.</param>
    /// <exception cref="NotSupportedException">No registered provider compiles for the profile.</exception>
    public static IExecutionProvider ForProfile(DeviceProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        List<Func<IExecutionProvider>> f;
        lock (s_lock)
        {
            f = [.. s_factories];
        }

        foreach (var factory in f)
        {
            IExecutionProvider provider;
            try
            {
                provider = factory();
            }
            catch (Exception)
            {
                // A provider that cannot even be constructed compiles nothing; Discover reports why.
                continue;
            }

            if (provider.CanCompileFor(profile))
            {
                return provider;
            }

            provider.Dispose();
        }

        throw new NotSupportedException($"No registered provider compiles for {profile}.");
    }
}
