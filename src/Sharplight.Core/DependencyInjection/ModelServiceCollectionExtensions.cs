using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Models;
using Sharplight.Vision;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers model formats and architectures: <see cref="IModelImporter"/>, <see cref="IModelAdapter"/>, <see cref="ICompositeModelAdapter"/>,
/// <see cref="IMultimodalArchitecture"/> and <see cref="AuxiliaryArchitecture"/> services, collected by the
/// <see cref="ModelImporterCatalog"/> and <see cref="ArchitectureCatalog"/> singletons that the <see cref="ModelLoader"/> uses.
/// Importer and architecture packages build on these (<c>services.AddGgufImporter()</c>, <c>services.AddBuiltinArchitectures()</c>).
/// </summary>
public static class ModelServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="ModelLoader"/> with its <see cref="ModelImporterCatalog"/> and <see cref="ArchitectureCatalog"/> (empty until
        /// importers and architectures are added). Idempotent; every registration below calls it.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddModelLoader()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton(provider => new ModelImporterCatalog(provider.GetServices<IModelImporter>()));
            services.TryAddSingleton(provider => new ArchitectureCatalog(provider.GetServices<IModelAdapter>(),
                provider.GetServices<IMultimodalArchitecture>(), provider.GetServices<AuxiliaryArchitecture>(), provider.GetServices<ICompositeModelAdapter>()));
            services.TryAddSingleton<ModelLoader>();
            return services;
        }

        /// <summary>Adds a model importer (a file format); a second registration of the same implementation type is ignored.</summary>
        /// <typeparam name="TImporter">The importer type, created by the container.</typeparam>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddModelImporter<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TImporter>()
            where TImporter : class, IModelImporter
        {
            services.AddModelLoader();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IModelImporter, TImporter>());
            return services;
        }

        /// <summary>Adds a model importer created by <paramref name="factory"/>; a second registration of the same type is ignored.</summary>
        /// <typeparam name="TImporter">The importer type.</typeparam>
        /// <param name="factory">Creates the importer (e.g. with options from the container).</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddModelImporter<TImporter>(Func<IServiceProvider, TImporter> factory)
            where TImporter : class, IModelImporter
        {
            ArgumentNullException.ThrowIfNull(factory);
            services.AddModelLoader();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IModelImporter, TImporter>(factory));
            return services;
        }

        /// <summary>Adds an architecture adapter; a second registration of the same implementation type is ignored.</summary>
        /// <typeparam name="TAdapter">The adapter type, created by the container.</typeparam>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddModelAdapter<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TAdapter>()
            where TAdapter : class, IModelAdapter
        {
            services.AddModelLoader();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IModelAdapter, TAdapter>());
            return services;
        }

        /// <summary>
        /// Adds an adapter that builds one model from several sources (<see cref="ICompositeModelAdapter"/>); a second registration of the
        /// same implementation type is ignored.
        /// </summary>
        /// <typeparam name="TAdapter">The adapter type, created by the container.</typeparam>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddCompositeModelAdapter<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TAdapter>()
            where TAdapter : class, ICompositeModelAdapter
        {
            services.AddModelLoader();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<ICompositeModelAdapter, TAdapter>());
            return services;
        }

        /// <summary>Adds the prompt assembly of image-capable architectures; a second registration of the same type is ignored.</summary>
        /// <typeparam name="TMultimodal">The handler type, created by the container.</typeparam>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddMultimodalArchitecture<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TMultimodal>()
            where TMultimodal : class, IMultimodalArchitecture
        {
            services.AddModelLoader();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IMultimodalArchitecture, TMultimodal>());
            return services;
        }

        /// <summary>
        /// Declares a canonical architecture that is only loaded as an auxiliary source of another model (see
        /// <see cref="AuxiliaryArchitecture"/>). Adding the same declaration twice has no effect.
        /// </summary>
        /// <param name="architecture">The canonical architecture id.</param>
        /// <param name="role">The auxiliary role its models are loaded in.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddAuxiliaryArchitecture(string architecture, string role)
        {
            ArgumentException.ThrowIfNullOrEmpty(architecture);
            ArgumentException.ThrowIfNullOrEmpty(role);
            services.AddModelLoader();
            var entry = new AuxiliaryArchitecture(architecture, role);
            if (!services.Any(d => d.ServiceType == typeof(AuxiliaryArchitecture) && Equals(d.ImplementationInstance, entry)))
            {
                services.AddSingleton(entry);
            }

            return services;
        }
    }
}
