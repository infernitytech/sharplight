using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Sharplight.Hosting;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers Sharplight hosting: the <see cref="IModelHostManager"/> (and its <see cref="ModelHost"/>), the <see cref="DeviceInventory"/>,
/// the <see cref="DefaultModelPlacementPolicy"/>, the chat, embedding, reranker and task loader factories and a hosted service that loads
/// the startup models. Needs the Sharplight services (<c>services.AddSharplight()</c>: model loader, execution providers, task layer).
/// </summary>
/// <example>
/// <code>
/// builder.Services.AddSharplight();
/// builder.Services.AddSharplightHosting(options =>
/// {
///     options.Reserve = 2L &lt;&lt; 30;
///     options.Models.Add(new HostedModelOptions { Name = "qwen3", Path = "/models/qwen3-4b.slm" });
///     options.Models.Add(new HostedModelOptions { Name = "bge", Path = "/models/bge-small.slm", Kind = ModelKind.Embedding, Lazy = true });
/// });
/// </code>
/// </example>
public static class HostingServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>Registers hosting; <paramref name="configure"/> adds the models and sets the policy. Idempotent (configurations add up).</summary>
        /// <param name="configure">Adjusts the <see cref="ModelHostingOptions"/>, or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightHosting(Action<ModelHostingOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<ModelHostingOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            services.TryAddSingleton<DeviceInventory>();
            services.AddRemoteSegments();
            services.TryAddSingleton<IModelPlacementPolicy, DefaultModelPlacementPolicy>();
            services.TryAddSingleton<ModelHostManager>();
            services.TryAddSingleton<IModelHostManager>(provider => provider.GetRequiredService<ModelHostManager>());
            services.TryAddSingleton(provider => provider.GetRequiredService<IModelHostManager>().Host);
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory, ChatModelLoaderFactory>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory, EmbeddingModelLoaderFactory>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory, RerankerModelLoaderFactory>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory, TaskModelLoaderFactory>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedService, HostedModelStartup>());
            return services;
        }

        /// <summary>Registers hosting with the models and policy of a JSON file (<see cref="ModelHostingJson"/>), then <paramref name="configure"/>.</summary>
        /// <param name="configurationFile">The hosting JSON file.</param>
        /// <param name="configure">Further adjustments (delegates such as logging), or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        /// <exception cref="FileNotFoundException">The file does not exist.</exception>
        /// <exception cref="FormatException">The file is not valid hosting options.</exception>
        public IServiceCollection AddSharplightHosting(string configurationFile, Action<ModelHostingOptions>? configure = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(configurationFile);
            var file = ModelHostingJson.Load(configurationFile);
            return services.AddSharplightHosting(options =>
            {
                ModelHostingJson.CopyTo(file, options);
                configure?.Invoke(options);
            });
        }
    }
}
