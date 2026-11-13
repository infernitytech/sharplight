using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>Optional capability of a mapping: one GGUF tensor that becomes several canonical weights (fused projections).</summary>
internal interface IGgufSplitMapping
{
    /// <summary>The canonical weights a GGUF tensor splits into, or null when the tensor maps one-to-one.</summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf);
}
