using Microsoft.Extensions.Options;
using Sharplight.Vulkan;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Vulkan execution provider.</summary>
public static class VulkanServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Vulkan execution provider (<see cref="VulkanProvider"/>) with <see cref="VulkanProviderOptions"/> from
        /// Microsoft.Extensions.Options. A machine without a Vulkan driver reports the provider unavailable. Idempotent; each call adds
        /// <paramref name="configure"/>.
        /// </summary>
        /// <param name="configure">Configures the provider options, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddVulkanProvider(Action<VulkanProviderOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<VulkanProviderOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services.AddExecutionProvider(provider => new VulkanProvider(provider.GetRequiredService<IOptions<VulkanProviderOptions>>().Value));
        }
    }
}
