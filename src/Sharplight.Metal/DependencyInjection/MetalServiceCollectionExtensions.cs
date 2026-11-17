using Microsoft.Extensions.Options;
using Sharplight.Metal;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Metal execution provider.</summary>
public static class MetalServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Metal execution provider (<see cref="MetalProvider"/>) with <see cref="MetalProviderOptions"/> from
        /// Microsoft.Extensions.Options. Off macOS, or without a Metal 3 GPU, the provider reports itself unavailable. Register it before the
        /// Vulkan provider so that discovery prefers it on Macs. Idempotent; each call adds <paramref name="configure"/>.
        /// </summary>
        /// <param name="configure">Configures the provider options, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddMetalProvider(Action<MetalProviderOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<MetalProviderOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services.AddExecutionProvider(provider => new MetalProvider(provider.GetRequiredService<IOptions<MetalProviderOptions>>().Value));
        }
    }
}
