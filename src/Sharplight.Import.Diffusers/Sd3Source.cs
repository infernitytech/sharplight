using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of an SD3 / SD3.5 model (architecture <c>sd3</c>): the transformer configuration
/// and whichever of the CLIP-L and CLIP-G text encoders, the T5 encoder (with its tokenizer) and the VAE the checkpoint brings. A
/// transformer-only source gets its companions as auxiliary sources.
/// </summary>
internal static class Sd3Source
{
    /// <summary>Canonical weight prefix of the CLIP-G encoder.</summary>
    public const string SecondTextEncoderPrefix = "text_encoder_2.";

    /// <summary>Canonical weight prefix of the T5 encoder.</summary>
    public const string T5Prefix = "text_encoder_3.";

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical weights (transformer, and the components present).</param>
    /// <param name="config">The transformer configuration.</param>
    /// <param name="resources">The open files backing the weights.</param>
    /// <param name="vae">The VAE configuration, or null when the source has no VAE.</param>
    /// <param name="clip">The CLIP-L configuration, or null.</param>
    /// <param name="clip2">The CLIP-G configuration, or null.</param>
    /// <param name="vocabulary">The CLIP tokenizer, or null for OpenAI's.</param>
    /// <param name="t5Metadata">The T5 encoder's canonical <c>encoder_decoder.*</c> and <c>tokenizer.*</c> keys, or null.</param>
    /// <param name="secondPadId">The padding id of CLIP-G prompts from the pipeline's second tokenizer, or null for SD3's <c>!</c>.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, Sd3Config config, IReadOnlyList<IDisposable> resources,
        VaeConfig? vae = null, ClipTextConfig? clip = null, ClipTextConfig? clip2 = null, ClipVocabulary? vocabulary = null,
        IReadOnlyDictionary<string, object>? t5Metadata = null, int? secondPadId = null)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = Sd3Config.Architecture,
        };
        if (vae is not null)
        {
            vae.WriteTo(metadata, Sd3Config.Prefix);
        }

        if (clip is not null)
        {
            clip.WriteTo(metadata, Sd3Config.Prefix, "text_encoder");
            (vocabulary ?? ClipVocabulary.OpenAi()).WriteTo(metadata, "", clip.ContextLength);
        }

        clip2?.WriteTo(metadata, Sd3Config.Prefix, "text_encoder_2");
        if (t5Metadata is not null)
        {
            foreach (var (key, value) in t5Metadata)
            {
                if (key.StartsWith("encoder_decoder.", StringComparison.Ordinal))
                {
                    metadata[Sd3Config.Prefix + "text_encoder_3." + key["encoder_decoder.".Length..]] = value;
                }
                else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                {
                    metadata[Sd3Config.Prefix + "tokenizer_3." + key] = value;
                }
            }
        }

        // The family keys (penultimate-layer CLIP conditioning, padding, prompt length, latent scaling) override what the parts say.
        config.WriteTo(metadata);
        if (secondPadId is { } pad)
        {
            metadata[Sd3Config.Prefix + "text_encoder_2.pad_id"] = pad;
        }

        return new ModelSource
        {
            Format = format,
            Architecture = Sd3Config.Architecture,
            Metadata = metadata,
            Weights = weights.Weights,
            OwnedResources = resources,
            Path = path,
        };
    }
}
