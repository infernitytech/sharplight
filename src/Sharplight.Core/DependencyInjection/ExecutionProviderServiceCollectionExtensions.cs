using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Runtime;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers execution providers: <see cref="IExecutionProvider"/> singletons, collected in registration order by the
/// <see cref="ExecutionProviderCatalog"/> singleton. Provider packages build on these (<c>services.AddVulkanProvider()</c>,
/// <c>services.AddCpuProvider()</c>).
/// </summary>
public static class ExecutionProviderServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>Adds an execution provider; a second registration of the same implementation type is ignored.</summary>
        /// <typeparam name="TProvider">The provider type, created by the container and disposed with it.</typeparam>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddExecutionProvider<TProvider>()
            where TProvider : class, IExecutionProvider
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton<ExecutionProviderCatalog>();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IExecutionProvider, TProvider>());
            return services;
        }

        /// <summary>Adds an execution provider created by <paramref name="factory"/>; a second registration of the same type is ignored.</summary>
        /// <typeparam name="TProvider">The provider type.</typeparam>
        /// <param name="factory">Creates the provider (e.g. with options from the container); the container disposes it.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddExecutionProvider<TProvider>(Func<IServiceProvider, TProvider> factory)
            where TProvider : class, IExecutionProvider
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(factory);
            services.TryAddSingleton<ExecutionProviderCatalog>();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IExecutionProvider, TProvider>(factory));
            return services;
        }
    }
}
