using Sharplight.Architectures.SeedVr;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the SeedVR2 architecture and its video-enhancement task client.</summary>
public static class SeedVrServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the SeedVR2 adapter (<see cref="SeedVrAdapter"/>) and the <c>IVideoEnhancer</c> implementation for its models; the checkpoints
        /// are read by <c>services.AddSeedVrImporter()</c>. Idempotent; <c>services.AddSharplight()</c> includes both.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSeedVrArchitecture()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddModelAdapter<SeedVrAdapter>().AddTaskClientFactory<SeedVrEnhancerFactory>();
        }
    }
}
