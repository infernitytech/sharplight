using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// Optional second stage of a mapping: rewrites the canonically named weight set where the file layout needs
/// several tensors combined into one canonical weight (derived lazily).
/// </summary>
internal interface IGgufWeightTransformer
{
    /// <summary>Rewrites <paramref name="weights"/> in place after all tensors have been renamed.</summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView gguf);
}
