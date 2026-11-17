using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Resolution;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the model feed services.</summary>
public static class ModelFeedServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the stateless feed services: <see cref="DependencyResolver"/>, <see cref="PackageDownloader"/>, <see cref="FeedPublisher"/> and
        /// an in-memory <see cref="IFeedStateStore"/> (replace it with a persistent one for rollback protection across restarts). Feed
        /// clients are created per feed with <see cref="FeedClient"/>. Idempotent.
        /// </summary>
        /// <returns>The services.</returns>
        public IServiceCollection AddModelFeed()
        {
            services.TryAddSingleton<DependencyResolver>();
            services.TryAddSingleton<PackageDownloader>();
            services.TryAddSingleton(provider => new FeedPublisher(provider.GetService<TimeProvider>()));
            services.TryAddSingleton<IFeedStateStore, MemoryFeedStateStore>();
            return services;
        }
    }
}
