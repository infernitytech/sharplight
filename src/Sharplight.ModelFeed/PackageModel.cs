namespace Sharplight.ModelFeed;

/// <summary>What a model is and what a host may offer it for.</summary>
public sealed record PackageModel
{
    /// <summary>Engine model kind (<c>text-generation</c>, <c>embeddings</c>, ...).</summary>
    public string? Kind { get; init; }

    /// <summary>Tasks (<see cref="PackageTasks"/>).</summary>
    public IReadOnlyList<string> Tasks { get; init; } = [];

    /// <summary>Capabilities (<see cref="PackageCapabilities"/>).</summary>
    public IReadOnlyList<string> Capabilities { get; init; } = [];

    /// <summary>Architecture name.</summary>
    public string? Architecture { get; init; }

    /// <summary>Parameter count.</summary>
    public long? Parameters { get; init; }

    /// <summary>Weight quantization (<c>Q8_0</c>, <c>F16</c>, ...).</summary>
    public string? Quantization { get; init; }

    /// <summary>Maximum context length in tokens.</summary>
    public int? ContextLength { get; init; }

    /// <summary>Embedding width for embedding models.</summary>
    public int? EmbeddingDimensions { get; init; }

    /// <summary>Languages (ISO 639-1).</summary>
    public IReadOnlyList<string> Languages { get; init; } = [];

    /// <summary>Input and output modalities.</summary>
    public PackageModalities? Modalities { get; init; }

    /// <summary>Knowledge cutoff date, when published.</summary>
    public string? KnowledgeCutoff { get; init; }

    /// <summary>Recommended sampling defaults.</summary>
    public PackageDefaults? Defaults { get; init; }
}
