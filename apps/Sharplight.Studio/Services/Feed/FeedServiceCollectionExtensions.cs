using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.ModelFeed.Resolution;
using Sharplight.ModelFeed.Versioning;
using Sharplight.Package;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Services.Feed;

/// <summary>Registers the model feed.</summary>
internal static class FeedServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Registers the model feed framework and <see cref="IFeedService"/>. The address and channel are read from <see cref="ISettingsService"/> on
        /// every use; needs <see cref="ISettingsService"/>, <see cref="StudioPaths"/> and <see cref="IUpdateService"/> (the host version) in the container.
        /// The trusted root key is the one the official feed is signed with (<c>tools/model-feed/keys/sharplight-feed.pub.pem</c>).
        /// </summary>
        public IServiceCollection AddStudioFeed()
        {
            services.AddModelFeed();
            services.TryAddSingleton(provider =>
            {
                var paths = provider.GetRequiredService<StudioPaths>();
                return new FeedOptions
                {
                    TrustedRoots = [FeedOptions.OfficialRootKey()],
                    StateFile = paths.FeedStateFile,
                    CacheDirectory = paths.FeedCacheDirectory,
                    Environment = new CompatibilityEnvironment
                    {
                        HostVersion = SemanticVersion.TryParse(provider.GetRequiredService<IUpdateService>().CurrentVersion, out var host) ? host : null,
                        EngineVersion = EngineVersion(),
                        MaxPackageFormat = ModelPackage.Version,
                    },
                };
            });
            services.TryAddSingleton<FeedService>();
            services.TryAddSingleton<IFeedService>(provider => provider.GetRequiredService<FeedService>());
            return services;
        }
    }

    private static SemanticVersion? EngineVersion()
    {
        var text = typeof(ModelPackage).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        return SemanticVersion.TryParse(text?.Split('+')[0], out var version) ? version : null;
    }
}
