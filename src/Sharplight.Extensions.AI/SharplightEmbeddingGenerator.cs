using Microsoft.Extensions.AI;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> over a Sharplight embedding model (<see cref="EmbeddingModel"/> or any
/// <see cref="ITextEmbedder"/>). <see cref="EmbeddingGenerationOptions.Dimensions"/> cuts Matryoshka embeddings (re-normalized);
/// <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> may carry <c>prompt_name</c> (one of the model's prompts, e.g.
/// "query") or <c>input_type</c> ("query" / "document": the model's prompt of that name when it has one) and <c>truncate</c>
/// (bool, default true). Inputs are embedded as one batch.
/// </summary>
public sealed class SharplightEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    readonly ITextEmbedder _model;
    readonly bool _ownsModel;
    readonly EmbeddingGeneratorMetadata _metadata;

    public SharplightEmbeddingGenerator(ITextEmbedder model, bool ownsModel = false, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _ownsModel = ownsModel;
        _metadata = new EmbeddingGeneratorMetadata("sharplight", null, modelId ?? model.Name, model.Dimensions);
    }

    /// <summary>Loads a model and returns a generator owning it (importers must be registered, e.g. <c>HuggingFaceImporter.Register()</c>).</summary>
    public static SharplightEmbeddingGenerator Load(string path, EmbeddingLoadOptions load)
        => new(EmbeddingModel.Load(path, load), ownsModel: true);

    public ITextEmbedder Model => _model;

    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        var texts = values.ToList();
        if (texts.Count == 0) return [];
        var o = ToEmbedOptions(options);
        var result = await Task.Run(() => _model.Embed(texts, o), cancellationToken).ConfigureAwait(false);
        var created = DateTimeOffset.UtcNow;
        string model = options?.ModelId ?? _metadata.DefaultModelId!;
        var list = new GeneratedEmbeddings<Embedding<float>>(result.Vectors.Select(v => new Embedding<float>(v) { ModelId = model, CreatedAt = created }))
        {
            Usage = new UsageDetails { InputTokenCount = result.TotalTokens, TotalTokenCount = result.TotalTokens },
        };
        return list;
    }

    EmbedOptions ToEmbedOptions(EmbeddingGenerationOptions? options)
    {
        var props = options?.AdditionalProperties;
        string? prompt = props?.TryGetValue("prompt_name", out var p) == true ? p?.ToString() : null;
        if (prompt is null && props?.TryGetValue("input_type", out var t) == true && t?.ToString() is { } type && _model.Prompts.ContainsKey(type)) prompt = type;
        bool truncate = props?.TryGetValue("truncate", out var tr) != true || tr is not bool b || b;
        return new EmbedOptions { Prompt = prompt, Dimensions = options?.Dimensions, Truncate = truncate };
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(EmbeddingGeneratorMetadata)) return _metadata;
        if (serviceType.IsInstanceOfType(_model)) return _model;
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        if (_ownsModel && _model is IDisposable d) d.Dispose();
    }
}
