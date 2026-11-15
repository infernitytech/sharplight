using Microsoft.Extensions.Options;
using Sharplight.AspNetCore;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Speech;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers the models an ASP.NET Core host serves and the <see cref="SharplightServerOptions"/> (Microsoft.Extensions.Options, so
/// they can also be bound from configuration). Pair with <c>app.UseSharplightServer()</c> and <c>app.MapSharplightServer()</c>.
/// </summary>
public static class SharplightServerServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Registers a single chat model and the server options. The model becomes the pinned default model, and requests naming
        /// an unknown model are answered by it (<see cref="SharplightServerOptions.UnknownModelFallsBackToDefault"/>).
        /// The caller keeps ownership of the engine behind <paramref name="model"/> (dispose it after the host stops).
        /// </summary>
        /// <param name="model">The chat service to expose.</param>
        /// <param name="configure">Adjusts the <see cref="SharplightServerOptions"/>, or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightServer(ModelService model, Action<SharplightServerOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            return services.AddServerContext(options => new SharplightServerContext(model, options), configure);
        }

        /// <summary>
        /// Registers a model host (several models, loaded on demand within the device memory budget) and the server options.
        /// The caller disposes the host after the server stops.
        /// </summary>
        /// <param name="host">The models to serve; clients select one with the <c>model</c> request field.</param>
        /// <param name="configure">Adjusts the <see cref="SharplightServerOptions"/>, or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightServer(ModelHost host, Action<SharplightServerOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(host);
            return services.AddServerContext(options => new SharplightServerContext(host, options), configure);
        }

        /// <summary>Serves <paramref name="embedder"/> on <c>/v1/embeddings</c> under <paramref name="name"/> (default: its own name).</summary>
        /// <param name="embedder">The embedding model; the caller keeps ownership.</param>
        /// <param name="name">The model name clients request, or <see langword="null"/> for the embedder's name.</param>
        /// <param name="configure">Adjusts the registered <see cref="RetrievalModels"/> (e.g. the default model), or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightEmbeddings(ITextEmbedder embedder, string? name = null, Action<RetrievalModels>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(embedder);
            var models = services.RetrievalModels();
            models.Embedders[name ?? embedder.Name] = embedder;
            models.DefaultEmbedder ??= name ?? embedder.Name;
            configure?.Invoke(models);
            return services;
        }

        /// <summary>Serves <paramref name="reranker"/> on <c>/v1/rerank</c> under <paramref name="name"/> (default: its own name).</summary>
        /// <param name="reranker">The reranker; the caller keeps ownership.</param>
        /// <param name="name">The model name clients request, or <see langword="null"/> for the reranker's name.</param>
        /// <param name="configure">Adjusts the registered <see cref="RetrievalModels"/> (e.g. the default model), or <see langword="null"/>.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightReranker(ITextReranker reranker, string? name = null, Action<RetrievalModels>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(reranker);
            var models = services.RetrievalModels();
            models.Rerankers[name ?? reranker.Name] = reranker;
            models.DefaultReranker ??= name ?? reranker.Name;
            configure?.Invoke(models);
            return services;
        }

        /// <summary>Serves <paramref name="recognizer"/> on the audio endpoints under <paramref name="name"/>.</summary>
        /// <param name="recognizer">The recognizer; the caller keeps ownership.</param>
        /// <param name="name">The model name clients pass.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightSpeechRecognizer(ISpeechRecognizer recognizer, string name)
        {
            ArgumentNullException.ThrowIfNull(recognizer);
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            var models = services.AudioModels();
            models.Recognizers[name] = recognizer;
            models.DefaultRecognizer ??= name;
            return services;
        }

        /// <summary>
        /// Serves the container's <see cref="ISpeechRecognizer"/> (registered e.g. with <c>services.AddSpeechRecognizer(...)</c>; keyed when
        /// <paramref name="serviceKey"/> is given) on the audio endpoints under <paramref name="name"/>. The model loads on the first request.
        /// </summary>
        /// <param name="name">The model name clients pass.</param>
        /// <param name="serviceKey">The recognizer's service key, or <see langword="null"/> for the unkeyed one.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightSpeechRecognizer(string name, object? serviceKey = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            var models = services.AudioModels();
            models.Lazy[name] = provider => serviceKey is null
                ? provider.GetRequiredService<ISpeechRecognizer>()
                : provider.GetRequiredKeyedService<ISpeechRecognizer>(serviceKey);
            models.DefaultRecognizer ??= name;
            return services;
        }

        private AudioModelRegistrations AudioModels()
        {
            if (services.FirstOrDefault(d => d.ServiceType == typeof(AudioModelRegistrations))?.ImplementationInstance is AudioModelRegistrations existing)
            {
                return existing;
            }

            var models = new AudioModelRegistrations();
            services.AddSingleton(models);
            services.AddSingleton(provider => models.Resolve(provider));
            return models;
        }
    

        private RetrievalModels RetrievalModels()
        {
            if (services.FirstOrDefault(d => d.ServiceType == typeof(RetrievalModels))?.ImplementationInstance is RetrievalModels existing)
            {
                return existing;
            }

            var models = new RetrievalModels();
            services.AddSingleton(models);
            return models;
        }

        private IServiceCollection AddServerContext(Func<SharplightServerOptions, SharplightServerContext> create, Action<SharplightServerOptions>? configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<SharplightServerOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            services.AddSingleton(provider => create(provider.GetRequiredService<IOptions<SharplightServerOptions>>().Value));
            services.AddHostedService(provider => new EngineLifetime(provider.GetRequiredService<SharplightServerContext>()));
            return services;
        }
    }
}
