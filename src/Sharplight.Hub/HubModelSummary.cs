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
public sealed record HubModelSummary(
    string Id,
    long Downloads,
    long Likes,
    bool Gated,
    string? Library,
    string? Pipeline,
    long? Parameters,
    IReadOnlyList<ModelFormat> Formats,
    IReadOnlyList<string> Variants);
