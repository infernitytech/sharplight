namespace Sharplight.Hub;

/// <summary>One search result.</summary>
/// <param name="Id">Repository id (<c>owner/name</c>).</param>
/// <param name="Downloads">Downloads in the last 30 days.</param>
/// <param name="Likes">Number of likes.</param>
/// <param name="Gated">Downloading requires accepting the repository's terms (and <c>HF_TOKEN</c>).</param>
/// <param name="Library">Library tag (e.g. <c>transformers</c>, <c>gguf</c>), or <see langword="null"/>.</param>
/// <param name="Pipeline">Pipeline tag (e.g. <c>text-generation</c>), or <see langword="null"/>.</param>
/// <param name="Parameters">Parameter count reported by the Hub (GGUF or safetensors metadata), or <see langword="null"/>.</param>
/// <param name="Formats">Formats the repository's files provide.</param>
/// <param name="Variants">Pullable variants (GGUF quantizations, ONNX model files); empty for a single checkpoint.</param>
/// <param name="GgufArchitecture">The <c>general.architecture</c> the Hub read from the repository's GGUF files, or <see langword="null"/>.</param>
/// <param name="ModelType">The <c>model_type</c> of the repository's <c>config.json</c> as the Hub reports it, or <see langword="null"/>.</param>
/// <param name="QuantizationMethod">
/// The <c>quantization_config.quant_method</c> of the repository's <c>config.json</c> as the Hub reports it (<c>mlx</c> for an MLX
/// configuration without a method), or <see langword="null"/>.
/// </param>
/// <param name="HasProjectorOnly">The repository's GGUF files are all multimodal projectors (mmproj).</param>
/// <param name="CreatedAt">When the repository was created, or <see langword="null"/> when the Hub did not say.</param>
public sealed record HubModelSummary(
    string Id,
    long Downloads,
    long Likes,
    bool Gated,
    string? Library,
    string? Pipeline,
    long? Parameters,
    IReadOnlyList<ModelFormat> Formats,
    IReadOnlyList<string> Variants,
    string? GgufArchitecture = null,
    string? ModelType = null,
    string? QuantizationMethod = null,
    bool HasProjectorOnly = false,
    DateTimeOffset? CreatedAt = null);
