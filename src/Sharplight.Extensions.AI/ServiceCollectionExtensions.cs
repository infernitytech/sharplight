using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Extensions.AI;

public static class SharplightServiceCollectionExtensions
{
    /// <summary>
    /// Registers a singleton <see cref="IChatClient"/> over the model at <paramref name="path"/> (loaded on first use; the
    /// container disposes it). Returns the <see cref="ChatClientBuilder"/> so middleware can be added:
    /// <c>services.AddSharplightChatClient(path, load).UseFunctionInvocation()</c>. The host must have registered
    /// importers (e.g. <c>GgufImporter.Register()</c>) and execution providers.
    /// </summary>
    public static ChatClientBuilder AddSharplightChatClient(this IServiceCollection services, string path,
        Func<IServiceProvider, LlmLoadOptions> load, Action<SharplightChatClientOptions>? configure = null)
        => services.AddChatClient(sp =>
        {
            var o = new SharplightChatClientOptions();
            configure?.Invoke(o);
            return SharplightChatClient.Load(path, load(sp), o);
        });

    /// <inheritdoc cref="AddSharplightChatClient(IServiceCollection, string, Func{IServiceProvider, LlmLoadOptions}, Action{SharplightChatClientOptions}?)"/>
    public static ChatClientBuilder AddSharplightChatClient(this IServiceCollection services, string path,
        LlmLoadOptions load, Action<SharplightChatClientOptions>? configure = null)
        => services.AddSharplightChatClient(path, _ => load, configure);

    /// <summary>Registers a client over an already loaded model (not disposed by the container).</summary>
    public static ChatClientBuilder AddSharplightChatClient(this IServiceCollection services, LlmModel model,
        Action<SharplightChatClientOptions>? configure = null)
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
    public static EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(this IServiceCollection services, string path,
        Func<IServiceProvider, EmbeddingLoadOptions> load)
        => services.AddEmbeddingGenerator(sp => SharplightEmbeddingGenerator.Load(path, load(sp)));

    /// <inheritdoc cref="AddSharplightEmbeddingGenerator(IServiceCollection, string, Func{IServiceProvider, EmbeddingLoadOptions})"/>
    public static EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(this IServiceCollection services, string path,
        EmbeddingLoadOptions load)
        => services.AddSharplightEmbeddingGenerator(path, _ => load);

    /// <summary>Registers a generator over an already loaded embedder (not disposed by the container).</summary>
    public static EmbeddingGeneratorBuilder<string, Embedding<float>> AddSharplightEmbeddingGenerator(this IServiceCollection services, ITextEmbedder model)
        => services.AddEmbeddingGenerator(_ => new SharplightEmbeddingGenerator(model));
}
