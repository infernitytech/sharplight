using Sharplight.Architectures.VideoEnhancement;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the video-enhancement architectures (SeedVR2) and their task clients.</summary>
public static class VideoEnhancementArchitectureServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the video-enhancement architectures of this package and their <c>IVideoEnhancer</c> implementations: SeedVR2
        /// (<see cref="SeedVrAdapter"/>; its checkpoints are read by <c>services.AddSeedVrImporter()</c>). Idempotent;
        /// <c>services.AddSharplight()</c> includes both.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddVideoEnhancementArchitectures()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddModelAdapter<SeedVrAdapter>().AddTaskClientFactory<SeedVrEnhancerFactory>();
        }
    }
}
