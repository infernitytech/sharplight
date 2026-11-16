using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of a Z-Image model (architecture <c>zimage</c>): the transformer configuration and
/// whichever of the text encoder (a decoder language model, kept whole under <c>text_encoder.</c> with its canonical metadata under
/// <c>zimage.text_encoder_source.</c>) and the VAE the checkpoint brings. A transformer-only source gets its companions as auxiliary sources.
/// </summary>
internal static class ZImageSource
{
    /// <summary>Canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    /// <summary>Metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = ZImageConfig.Prefix + "text_encoder_source.";

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical transformer (and VAE) weights.</param>
    /// <param name="config">The transformer configuration.</param>
    /// <param name="resources">The open files backing the weights.</param>
    /// <param name="vae">The VAE configuration, or null when the source has no VAE.</param>
    /// <param name="textEncoder">The text encoder as a decoder source (its resources become this source's), or null.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, ZImageConfig config, IReadOnlyList<IDisposable> resources,
        VaeConfig? vae = null, ModelSource? textEncoder = null)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = ZImageConfig.Architecture,
        };
        vae?.WriteTo(metadata, ZImageConfig.Prefix);
        var allWeights = new Dictionary<string, WeightInfo>(weights.Weights, StringComparer.Ordinal);
        var allResources = new List<IDisposable>(resources);
        if (textEncoder is not null)
        {
            foreach (var (key, value) in textEncoder.Metadata)
            {
                metadata[TextEncoderSourcePrefix + key] = value;
                if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                {
                    metadata[key] = value;
                }
            }

            metadata[TextEncoderSourcePrefix + "architecture"] = textEncoder.Architecture;
            foreach (var (weightName, weight) in textEncoder.Weights)
            {
                allWeights[TextEncoderPrefix + weightName] = weight.As(TextEncoderPrefix + weightName);
            }

            allResources.AddRange(textEncoder.OwnedResources);
        }

        // The family keys (prompt template and length, latent scaling) override what the parts say.
        config.WriteTo(metadata);
        return new ModelSource
        {
            Format = format,
            Architecture = ZImageConfig.Architecture,
            Metadata = metadata,
            Weights = allWeights,
            OwnedResources = allResources,
            Path = path,
        };
    }
}
