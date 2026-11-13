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

    private sealed class UnavailableProvider(string name, string reason) : IExecutionProvider
    {
        public string Name => name;
        public ProviderAvailability CheckAvailability() => ProviderAvailability.Unavailable(reason);
        public IReadOnlyList<DeviceInfo> GetDevices() => [];
        public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null) => throw new NotSupportedException(reason);
        public void Dispose() { }
    }
}
