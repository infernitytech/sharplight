using Microsoft.Extensions.AI;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> over a Sharplight text embedder (<see cref="ITextEmbedder"/>, e.g. from
/// <c>services.AddTextEmbedder(...)</c>). <see cref="EmbeddingGenerationOptions.Dimensions"/> cuts Matryoshka embeddings (re-normalized);
/// <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> may carry <c>prompt_name</c> (one of the model's prompts, e.g.
/// "query") or <c>input_type</c> ("query" / "document": the model's prompt of that name when it has one) and <c>truncate</c>
/// (bool, default true). Inputs are embedded as one batch.
/// </summary>
public sealed class SharplightEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly ITextEmbedder _model;
    private readonly bool _ownsModel;
    private readonly EmbeddingGeneratorMetadata _metadata;

    /// <summary>Wraps <paramref name="model"/>; disposes it with the generator when <paramref name="ownsModel"/>. <paramref name="modelId"/> overrides the reported model id.</summary>
    public SharplightEmbeddingGenerator(ITextEmbedder model, bool ownsModel = false, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _ownsModel = ownsModel;
        _metadata = new EmbeddingGeneratorMetadata("sharplight", null, modelId ?? model.Metadata.ModelName, model.Dimensions);
    }

    /// <summary>The wrapped embedder.</summary>
    public ITextEmbedder Model => _model;

    /// <inheritdoc/>
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        var texts = values.ToList();
        if (texts.Count == 0)
        {
            return [];
        }

        var o = ToEmbedOptions(options);
        var result = await _model.EmbedAsync(texts, o, cancellationToken).ConfigureAwait(false);
        var created = DateTimeOffset.UtcNow;
        string model = options?.ModelId ?? _metadata.DefaultModelId!;
        var list = new GeneratedEmbeddings<Embedding<float>>(result.Vectors.Select(v => new Embedding<float>(v) { ModelId = model, CreatedAt = created }))
        {
            Usage = new UsageDetails { InputTokenCount = result.TotalTokens, TotalTokenCount = result.TotalTokens },
        };
        return list;
    }

    private TextEmbeddingOptions ToEmbedOptions(EmbeddingGenerationOptions? options)
    {
        var props = options?.AdditionalProperties;
        string? prompt = props?.TryGetValue("prompt_name", out var p) == true ? p?.ToString() : null;
        if (prompt is null && props?.TryGetValue("input_type", out var t) == true && t?.ToString() is { } type && _model.Prompts.ContainsKey(type))
        {
            prompt = type;
        }

        bool truncate = props?.TryGetValue("truncate", out var tr) != true || tr is not bool b || b;
        return new TextEmbeddingOptions { Prompt = prompt, Dimensions = options?.Dimensions, Truncate = truncate };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(EmbeddingGeneratorMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(_model))
        {
            return _model;
        }

        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsModel && _model is IDisposable d)
        {
            d.Dispose();
        }
    }
}
