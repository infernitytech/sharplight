using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Storage;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the <see cref="StorageLayout"/>: where Sharplight writes configuration, data, models, caches and logs.</summary>
public static class StorageServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="StorageLayout"/> of this process (<see cref="StorageLayout.Resolve"/> with
        /// <see cref="StorageEnvironment.Process"/>) unless one is registered already. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightStorage()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton(_ => StorageLayout.Resolve(StorageEnvironment.Process));
            return services;
        }

        /// <summary>Adds <paramref name="layout"/> as the <see cref="StorageLayout"/> unless one is registered already. Idempotent.</summary>
        /// <param name="layout">A layout the host resolved itself (e.g. before its settings were loaded).</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightStorage(StorageLayout layout)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(layout);
            services.TryAddSingleton(layout);
            return services;
        }
    }
}
