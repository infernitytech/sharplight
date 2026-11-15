namespace Sharplight.Llm.Embeddings;

/// <summary>Options for <see cref="EmbeddingModel.Load"/>.</summary>
public sealed class EmbeddingLoadOptions : PackedModelLoadOptions
{
    /// <summary>Pooling override ("mean", "cls", "last", "none"); null = what the model declares.</summary>
    public string? Pooling { get; init; }
    /// <summary>L2 normalization override; null = what the model declares.</summary>
    public bool? Normalize { get; init; }
}
