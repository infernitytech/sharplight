using Microsoft.Extensions.DependencyInjection;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Registers the execution providers that only some Studio packages carry.</summary>
internal static class StudioProviderExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the CUDA provider after the Vulkan and CPU ones (so Vulkan stays the first GPU provider) in the packages that ship it:
        /// linux-x64, linux-arm64 and win-x64 (<c>StudioCuda</c> in the project file); a no-op in the others. Without an NVIDIA driver
        /// or GPU the provider reports itself unavailable and the device list shows a note.
        /// </summary>
        public IServiceCollection AddStudioProviders()
        {
#if STUDIO_CUDA
            services.AddCudaProvider();
#endif
            return services;
        }
    }
}
