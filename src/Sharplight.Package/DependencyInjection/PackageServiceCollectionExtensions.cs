using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Package;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the <c>.slm</c> package format.</summary>
public static class PackageServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <c>.slm</c> package importer (<see cref="ModelPackageImporter"/>), so the model loader opens saved packages (their
        /// prebuilt definitions need no architecture adapter), and the <see cref="ModelConverter"/>. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddPackageImporter()
        {
            services.AddModelImporter<ModelPackageImporter>();
            services.TryAddSingleton<ModelConverter>();
            return services;
        }
    }
}
