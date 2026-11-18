using Microsoft.Extensions.Options;
using Sharplight.Cuda;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the CUDA execution provider.</summary>
public static class CudaServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the CUDA execution provider (<see cref="CudaProvider"/>) with <see cref="CudaProviderOptions"/> from
        /// Microsoft.Extensions.Options. Without the NVIDIA driver, or without a supported GPU, the provider reports itself unavailable.
        /// Providers are listed in registration order: register it after the Vulkan provider to keep Vulkan the default GPU provider
        /// (the CUDA provider is then used when a host picks it, e.g. <c>--provider cuda</c>), or before it to prefer CUDA on NVIDIA GPUs.
        /// Idempotent; each call adds <paramref name="configure"/>.
        /// </summary>
        /// <param name="configure">Configures the provider options, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddCudaProvider(Action<CudaProviderOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<CudaProviderOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services.AddExecutionProvider(provider => new CudaProvider(provider.GetRequiredService<IOptions<CudaProviderOptions>>().Value));
        }
    }
}
