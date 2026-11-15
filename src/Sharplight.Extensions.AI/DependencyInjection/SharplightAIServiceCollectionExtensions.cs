using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using Sharplight.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Speech;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Dependency-injection registration of the Microsoft.Extensions.AI adapters over Sharplight models and task clients.</summary>
public static class SharplightAIServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Registers a singleton <see cref="IChatClient"/> over one model, configured like the task clients: the model is loaded on first
        /// use through the container's <see cref="ModelLoader"/> with the <see cref="LlmLoadOptions"/> named after
        /// <paramref name="serviceKey"/> (configured by <paramref name="configure"/> and any other options configuration, e.g. bound from
        /// <c>IConfiguration</c>), and the container disposes it. Returns the <see cref="ChatClientBuilder"/> so middleware can be added.
        /// </summary>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="client">Configures the <see cref="SharplightChatClientOptions"/>, or <see langword="null"/> for the defaults.</param>
        /// <param name="serviceKey">The service key of the client, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The chat client builder.</returns>
        /// <example>
        /// <code>
        /// services.AddSharplight();
        /// services.AddSharplightChatClient(o => { o.ModelPath = "qwen3-4b-q4_k_m.gguf"; o.Devices = [slot]; })
        ///     .UseFunctionInvocation();
        /// var chat = services.BuildServiceProvider().GetRequiredService&lt;IChatClient&gt;();
        /// </code>
        /// </example>
        public ChatClientBuilder AddSharplightChatClient(Action<LlmLoadOptions> configure, Action<SharplightChatClientOptions>? client = null,
            object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(configure);
            return services.AddSharplightChatClient((options, _) => configure(options), client, serviceKey);
        }

        /// <summary>
        /// Registers a singleton <see cref="IChatClient"/> over one model (see
        /// <see cref="AddSharplightChatClient(Action{LlmLoadOptions}, Action{SharplightChatClientOptions}?, object?)"/>), configured with
        /// access to the container, e.g. to open a device from the <see cref="Sharplight.Runtime.ExecutionProviderCatalog"/>.
        /// </summary>
        /// <param name="configure">Configures the model path, devices and compile settings from the container's services.</param>
        /// <param name="client">Configures the <see cref="SharplightChatClientOptions"/>, or <see langword="null"/> for the defaults.</param>
        /// <param name="serviceKey">The service key of the client, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The chat client builder.</returns>
        public ChatClientBuilder AddSharplightChatClient(Action<LlmLoadOptions, IServiceProvider> configure, Action<SharplightChatClientOptions>? client = null,
            object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            string name = OptionsName(typeof(IChatClient), serviceKey);
            services.AddModelLoader();
            services.AddOptions<LlmLoadOptions>(name).Configure<IServiceProvider>(configure);
            var clientOptions = services.AddOptions<SharplightChatClientOptions>(name);
            if (client is not null)
            {
                clientOptions.Configure(client);
            }

            IChatClient Create(IServiceProvider provider)
            {
                var load = provider.GetRequiredService<IOptionsMonitor<LlmLoadOptions>>().Get(name);
                var options = provider.GetRequiredService<IOptionsMonitor<SharplightChatClientOptions>>().Get(name);
                return SharplightChatClient.Load(provider.GetRequiredService<ModelLoader>(), RequirePath(load.ModelPath), load, options);
            }

            return serviceKey is null ? services.AddChatClient(Create) : services.AddKeyedChatClient(serviceKey, Create);
        }

        /// <summary>Registers a client over an already loaded model (not disposed by the container).</summary>
        /// <param name="model">The loaded model.</param>
        /// <param name="configure">Adjusts the <see cref="SharplightChatClientOptions"/>, or <see langword="null"/>.</param>
        /// <returns>The chat client builder.</returns>
        public ChatClientBuilder AddSharplightChatClient(LlmModel model, Action<SharplightChatClientOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            return services.AddChatClient(_ =>
            {
                var options = new SharplightChatClientOptions();
                configure?.Invoke(options);
                return new SharplightChatClient(model, options);
            });
        }

        /// <summary>
        /// Registers a singleton <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> over one embedding model, loaded on first use with
        /// the <see cref="EmbeddingLoadOptions"/> named after <paramref name="serviceKey"/> (see
        /// <see cref="AddSharplightChatClient(Action{LlmLoadOptions}, Action{SharplightChatClientOptions}?, object?)"/>). Returns the
        /// builder for middleware (e.g. <c>.UseDistributedCache()</c>).
        /// </summary>
        /// <param name="configure">Configures the model path, devices and batch settings.</param>
        /// <param name="serviceKey">The service key of the generator, or <see langword="null"/> for an unkeyed generator.</param>
        /// <returns>The embedding generator builder.</returns>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(Action<EmbeddingLoadOptions> configure, object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(configure);
            return services.AddSharplightEmbeddingGenerator((options, _) => configure(options), serviceKey);
        }

        /// <summary>
        /// Registers a singleton embedding generator over one model (see
        /// <see cref="AddSharplightEmbeddingGenerator(Action{EmbeddingLoadOptions}, object?)"/>), configured with access to the container.
        /// </summary>
        /// <param name="configure">Configures the model path, devices and batch settings from the container's services.</param>
        /// <param name="serviceKey">The service key of the generator, or <see langword="null"/> for an unkeyed generator.</param>
        /// <returns>The embedding generator builder.</returns>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(Action<EmbeddingLoadOptions, IServiceProvider> configure,
            object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            string name = OptionsName(typeof(IEmbeddingGenerator<string, Embedding<float>>), serviceKey);
            services.AddModelLoader();
            services.AddOptions<EmbeddingLoadOptions>(name).Configure<IServiceProvider>(configure);

            IEmbeddingGenerator<string, Embedding<float>> Create(IServiceProvider provider)
            {
                var load = provider.GetRequiredService<IOptionsMonitor<EmbeddingLoadOptions>>().Get(name);
                return SharplightEmbeddingGenerator.Load(provider.GetRequiredService<ModelLoader>(), RequirePath(load.ModelPath), load);
            }

            return serviceKey is null ? services.AddEmbeddingGenerator(Create) : services.AddKeyedEmbeddingGenerator(serviceKey, Create);
        }

        /// <summary>Registers a generator over an already loaded embedder (not disposed by the container).</summary>
        /// <param name="model">The loaded embedder.</param>
        /// <returns>The embedding generator builder.</returns>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(ITextEmbedder model)
        {
            ArgumentNullException.ThrowIfNull(model);
            return services.AddEmbeddingGenerator(_ => new SharplightEmbeddingGenerator(model));
        }

        /// <summary>
        /// Registers an <see cref="ISpeechToTextClient"/> over the container's <see cref="ISpeechRecognizer"/> (register one with
        /// <c>services.AddSpeechRecognizer(...)</c>); with <paramref name="serviceKey"/>, both are keyed services of that key.
        /// </summary>
        /// <param name="serviceKey">The service key of the recognizer and the client, or <see langword="null"/>.</param>
        public IServiceCollection AddSharplightSpeechToTextClient(object? serviceKey = null)
            => services.AddAdapter<ISpeechToTextClient, ISpeechRecognizer>(serviceKey, recognizer => new SharplightSpeechToTextClient(recognizer));

        /// <summary>
        /// Registers an <see cref="ITextToSpeechClient"/> over the container's <see cref="ISpeechSynthesizer"/> (register one with
        /// <c>services.AddSpeechSynthesizer(...)</c>); with <paramref name="serviceKey"/>, both are keyed services of that key.
        /// </summary>
        /// <param name="serviceKey">The service key of the synthesizer and the client, or <see langword="null"/>.</param>
        public IServiceCollection AddSharplightTextToSpeechClient(object? serviceKey = null)
            => services.AddAdapter<ITextToSpeechClient, ISpeechSynthesizer>(serviceKey, synthesizer => new SharplightTextToSpeechClient(synthesizer));

        /// <summary>
        /// Registers a Microsoft.Extensions.AI <see cref="IImageGenerator"/> over the container's Sharplight image generator (register one
        /// with <c>services.AddImageGenerator(...)</c>); with <paramref name="serviceKey"/>, both are keyed services of that key.
        /// </summary>
        /// <param name="serviceKey">The service key of the Sharplight generator and the adapter, or <see langword="null"/>.</param>
        public IServiceCollection AddSharplightImageGenerator(object? serviceKey = null)
            => services.AddAdapter<IImageGenerator, Sharplight.ImageGeneration.IImageGenerator>(serviceKey, generator => new SharplightImageGenerator(generator));

        private IServiceCollection AddAdapter<TAdapter, TClient>(object? serviceKey, Func<TClient, TAdapter> create)
            where TAdapter : class
            where TClient : class
        {
            ArgumentNullException.ThrowIfNull(services);
            if (serviceKey is null)
            {
                return services.AddSingleton(provider => create(provider.GetRequiredService<TClient>()));
            }

            return services.AddKeyedSingleton(serviceKey, (provider, key) => create(provider.GetRequiredKeyedService<TClient>(key)));
        }
    }

    private static string OptionsName(Type client, object? serviceKey) => serviceKey is null ? client.Name : $"{client.Name}:{serviceKey}";

    private static string RequirePath(string? path)
        => string.IsNullOrEmpty(path) ? throw new InvalidOperationException("The load options have no ModelPath: set it in the configure callback.") : path;
}
