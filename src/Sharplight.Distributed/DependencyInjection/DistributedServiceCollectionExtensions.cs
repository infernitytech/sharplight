using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Distributed;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers multi-node inference (remote segment workers).</summary>
public static class DistributedServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="RemoteDevicePool"/> through which hosts open segment workers as devices (hosting's per-model <c>workers</c>),
        /// with <see cref="RemoteSegmentOptions"/> from the options pattern (transport and TLS, heartbeats, timeouts, weight verification).
        /// The remote provider is not added to the execution provider catalog: automatic device selection never picks a remote worker.
        /// Idempotent.
        /// </summary>
        /// <param name="configure">Adjusts the options, or null.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddRemoteSegments(Action<RemoteSegmentOptions>? configure = null)
        {
            services.AddOptions<RemoteSegmentOptions>();
            if (configure is not null)
            {
                services.Configure(configure);
            }

            services.TryAddSingleton<RemoteDevicePool>();
            return services;
        }
    }
}
