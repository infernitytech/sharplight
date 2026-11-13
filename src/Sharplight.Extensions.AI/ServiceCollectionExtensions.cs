using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Extensions.AI;

/// <summary>Dependency-injection registration of Sharplight chat clients and embedding generators.</summary>
public static class SharplightServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Registers a singleton <see cref="IChatClient"/> over the model at <paramref name="path"/> (loaded on first use; the
        /// container disposes it). Returns the <see cref="ChatClientBuilder"/> so middleware can be added:
        /// <c>services.AddSharplightChatClient(path, load).UseFunctionInvocation()</c>. The host must have registered
        /// importers (e.g. <c>GgufImporter.Register()</c>) and execution providers.
        /// </summary>
        /// <param name="path">Model file or directory.</param>
        /// <param name="load">Creates the load options (devices, context length, ...) from the service provider.</param>
        /// <param name="configure">Adjusts the <see cref="SharplightChatClientOptions"/>.</param>
        /// <example>
        /// <code>
        /// GgufImporter.Register();
        /// var services = new ServiceCollection();
        /// services.AddSharplightChatClient("qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [device] })
        ///     .UseFunctionInvocation();
        /// var chat = services.BuildServiceProvider().GetRequiredService&lt;IChatClient&gt;();
        /// </code>
        /// </example>
        public ChatClientBuilder AddSharplightChatClient(string path, Func<IServiceProvider, LlmLoadOptions> load,
            Action<SharplightChatClientOptions>? configure = null)
            => services.AddChatClient(sp =>
            {
                var o = new SharplightChatClientOptions();
                configure?.Invoke(o);
                return SharplightChatClient.Load(path, load(sp), o);
            });

        /// <summary>Registers a singleton <see cref="IChatClient"/> over the model at <paramref name="path"/> loaded with <paramref name="load"/>.</summary>
        public ChatClientBuilder AddSharplightChatClient(string path, LlmLoadOptions load, Action<SharplightChatClientOptions>? configure = null)
            => services.AddSharplightChatClient(path, _ => load, configure);

        /// <summary>Registers a client over an already loaded model (not disposed by the container).</summary>
        public ChatClientBuilder AddSharplightChatClient(LlmModel model, Action<SharplightChatClientOptions>? configure = null)
            => services.AddChatClient(_ =>
            {
                var o = new SharplightChatClientOptions();
                configure?.Invoke(o);
                return new SharplightChatClient(model, o);
            });

        /// <summary>
        /// Registers a singleton <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> over the embedding model at <paramref name="path"/>
        /// (loaded on first use; the container disposes it). Returns the builder for middleware (e.g. <c>.UseDistributedCache()</c>).
        /// </summary>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(string path, Func<IServiceProvider, EmbeddingLoadOptions> load)
            => services.AddEmbeddingGenerator(sp => SharplightEmbeddingGenerator.Load(path, load(sp)));

        /// <summary>Registers a singleton embedding generator over the model at <paramref name="path"/> loaded with <paramref name="load"/>.</summary>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(string path, EmbeddingLoadOptions load)
            => services.AddSharplightEmbeddingGenerator(path, _ => load);

        /// <summary>Registers a generator over an already loaded embedder (not disposed by the container).</summary>
        public EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(ITextEmbedder model)
            => services.AddEmbeddingGenerator(_ => new SharplightEmbeddingGenerator(model));
    }
}
