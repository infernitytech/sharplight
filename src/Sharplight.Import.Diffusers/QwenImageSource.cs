using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of a Qwen-Image model (architecture <c>qwen_image</c>): the transformer configuration
/// and whichever of the text encoder (a decoder language model, Qwen2.5-VL's text model, kept whole under <c>text_encoder.</c> with its
/// canonical metadata under <c>qwen_image.text_encoder_source.</c>) and the causal 3-D VAE (<c>vae.*</c>, configuration under
/// <c>qwen_image.vae.*</c>) the checkpoint brings. A transformer-only source gets its companions as auxiliary sources.
/// </summary>
internal static class QwenImageSource
{
    /// <summary>Canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    /// <summary>Metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = QwenImageConfig.Prefix + "text_encoder_source.";

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical transformer (and VAE) weights.</param>
    /// <param name="config">The transformer configuration.</param>
    /// <param name="resources">The open files backing the weights.</param>
    /// <param name="vaeMetadata">The VAE keys (<c>qwen_image.vae.*</c>), or null when the source has no VAE.</param>
    /// <param name="textEncoder">The text encoder as a decoder source (its resources become this source's), or null.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, QwenImageConfig config, IReadOnlyList<IDisposable> resources,
        IReadOnlyDictionary<string, object>? vaeMetadata = null, ModelSource? textEncoder = null)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = QwenImageConfig.Architecture,
        };
        foreach (var (key, value) in vaeMetadata ?? new Dictionary<string, object>())
        {
            metadata[key] = value;
        }

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

        // The family keys (prompt template, skip and length, scheduler) override what the parts say.
        config.WriteTo(metadata);
        return new ModelSource
        {
            Format = format,
            Architecture = QwenImageConfig.Architecture,
            Metadata = metadata,
            Weights = allWeights,
            OwnedResources = allResources,
            Path = path,
        };
    }
}
