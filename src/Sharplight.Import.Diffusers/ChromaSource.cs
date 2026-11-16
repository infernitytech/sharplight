using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of a Chroma model (architecture <c>chroma</c>): the transformer configuration and
/// whichever of the T5 encoder (with its tokenizer) and the VAE the checkpoint brings. A transformer-only source gets its companions as
/// auxiliary sources.
/// </summary>
internal static class ChromaSource
{
    /// <summary>Canonical weight prefix of the T5 encoder.</summary>
    public const string T5Prefix = "text_encoder.";

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="config">The transformer configuration.</param>
    /// <param name="resources">The open files backing the weights.</param>
    /// <param name="vae">The VAE configuration, or null when the source has no VAE.</param>
    /// <param name="t5Metadata">The T5 encoder's canonical <c>encoder_decoder.*</c> and <c>tokenizer.*</c> keys, or null.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, ChromaConfig config, IReadOnlyList<IDisposable> resources,
        VaeConfig? vae = null, IReadOnlyDictionary<string, object>? t5Metadata = null)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = ChromaConfig.Architecture,
        };
        vae?.WriteTo(metadata, ChromaConfig.Prefix);
        foreach (var (key, value) in t5Metadata ?? new Dictionary<string, object>())
        {
            if (key.StartsWith("encoder_decoder.", StringComparison.Ordinal))
            {
                metadata[ChromaConfig.Prefix + "text_encoder." + key["encoder_decoder.".Length..]] = value;
            }
            else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
            {
                metadata[key] = value;
            }
        }

        // The family keys (prompt length, latent scaling) override what the parts say.
        config.WriteTo(metadata);
        return new ModelSource
        {
            Format = format,
            Architecture = ChromaConfig.Architecture,
            Metadata = metadata,
            Weights = weights.Weights,
            OwnedResources = resources,
            Path = path,
        };
    }
}
