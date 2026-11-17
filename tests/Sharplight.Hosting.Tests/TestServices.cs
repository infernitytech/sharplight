using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Hosting.Loaders;
using Sharplight.Runtime;

namespace Sharplight.Hosting.Tests;

/// <summary>A container with hosting over fake models on one fake pool (no execution provider is touched).</summary>
internal static class TestServices
{
    public static ServiceProvider Create(FakeLoaderFactory factory, Action<ModelHostingOptions> configure)
    {
        var services = new ServiceCollection();
        services.AddSingleton(new ExecutionProviderCatalog([]));
        services.AddSharplightHosting(configure);
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory>(factory));
        return services.BuildServiceProvider();
    }
}
