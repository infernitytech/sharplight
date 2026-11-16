using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The decoder language model inside an image generator's source (Z-Image's Qwen3, Qwen-Image's Qwen2.5-VL): its canonical metadata
/// kept under a metadata prefix and its weights under a weight prefix, taken out again as a decoder source of its own for
/// <see cref="DecoderGraphBuilder"/>.
/// </summary>
internal static class DecoderTextEncoder
{
    /// <summary>The text encoder as a decoder source: its canonical metadata and its weights without the prefixes.</summary>
    /// <param name="src">The image generator's source.</param>
    /// <param name="metadataPrefix">The prefix of the encoder's metadata (e.g. <c>zimage.text_encoder_source.</c>).</param>
    /// <param name="weightPrefix">The prefix of the encoder's weights (<c>text_encoder.</c>).</param>
    public static ModelSource Source(ModelSource src, string metadataPrefix, string weightPrefix)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in src.Metadata)
        {
            if (key.StartsWith(metadataPrefix, StringComparison.Ordinal))
            {
                metadata[key[metadataPrefix.Length..]] = value;
            }
        }

        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, weight) in src.Weights)
        {
            if (name.StartsWith(weightPrefix, StringComparison.Ordinal))
            {
                weights[name[weightPrefix.Length..]] = weight;
            }
        }

        string architecture = metadata.TryGetValue("architecture", out var stored) ? stored.ToString()! : "decoder";
        return new ModelSource
        {
            Format = src.Format,
            Architecture = architecture,
            Metadata = metadata,
            Weights = weights,
            Path = src.Path,
        };
    }
}
