using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Updates;

/// <summary>Registers the update service.</summary>
internal static class UpdateServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Registers <see cref="IUpdateService"/> (Velopack). The feed URL comes from <see cref="UpdateEndpoint"/>; it needs an
        /// <see cref="ISettingsService"/> in the container for the <c>UpdateUrl</c> setting.
        /// </summary>
        public IServiceCollection AddStudioUpdates()
        {
            services.TryAddSingleton<IUpdateService>(provider => new VelopackUpdateService(UpdateEndpoint.Resolve(
                Environment.GetEnvironmentVariable(UpdateEndpoint.EnvironmentVariable),
                provider.GetRequiredService<ISettingsService>().Current.UpdateUrl,
                UpdateEndpoint.Baked(typeof(UpdateEndpoint).Assembly))));
            return services;
        }
    }
}
