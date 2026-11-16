using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// A language model embedded in a composite source (an image or video generator's text encoder) as a canonical source of its own: the
/// metadata under a prefix without it (its architecture under <c>{prefix}architecture</c>) and the weights under a prefix without it.
/// </summary>
internal static class DecoderSubSource
{
    /// <summary>Extracts the embedded source.</summary>
    /// <param name="src">The composite source.</param>
    /// <param name="metadataPrefix">The metadata prefix of the embedded source (<c>zimage.text_encoder_source.</c>).</param>
    /// <param name="weightPrefix">The weight prefix of the embedded source (<c>text_encoder.</c>).</param>
    public static ModelSource Extract(ModelSource src, string metadataPrefix, string weightPrefix)
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
