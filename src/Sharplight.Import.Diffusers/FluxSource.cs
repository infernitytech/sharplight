using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of a FLUX model (architecture <c>flux</c>): the transformer configuration, and
/// whichever of the CLIP text encoder, the T5 encoder (with its tokenizer) and the VAE the checkpoint brings. A transformer-only
/// source gets its companions as auxiliary sources (see <see cref="DiffusionCompanions"/>).
/// </summary>
internal static class FluxSource
{
    /// <summary>Canonical weight prefix of the T5 encoder.</summary>
    public const string T5Prefix = "text_encoder_2.";

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical weights (transformer, and the components present).</param>
    /// <param name="config">The transformer configuration.</param>
    /// <param name="resources">The open files backing the weights.</param>
    /// <param name="vae">The VAE configuration, or null when the source has no VAE.</param>
    /// <param name="clip">The CLIP text encoder configuration, or null.</param>
    /// <param name="vocabulary">The CLIP tokenizer, or null.</param>
    /// <param name="t5Metadata">The T5 encoder's canonical <c>encoder_decoder.*</c> and <c>tokenizer.*</c> keys, or null.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, FluxConfig config, IReadOnlyList<IDisposable> resources,
        VaeConfig? vae = null, ClipTextConfig? clip = null, ClipVocabulary? vocabulary = null, IReadOnlyDictionary<string, object>? t5Metadata = null)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = FluxConfig.Architecture,
        };
        config.WriteTo(metadata);
        if (vae is not null)
        {
            vae.WriteTo(metadata, FluxConfig.Prefix);
        }

        if (clip is not null)
        {
            clip.WriteTo(metadata, FluxConfig.Prefix, "text_encoder");
            (vocabulary ?? ClipVocabulary.OpenAi()).WriteTo(metadata, "", clip.ContextLength);
        }

        if (t5Metadata is not null)
        {
            AddT5Metadata(metadata, t5Metadata);
        }

        return new ModelSource
        {
            Format = format,
            Architecture = FluxConfig.Architecture,
            Metadata = metadata,
            Weights = weights.Weights,
            OwnedResources = resources,
            Path = path,
        };
    }

    /// <summary>
    /// The T5 encoder of a diffusers folder (<c>text_encoder_2/</c> with <c>tokenizer_2/</c>): its weights in the encoder vocabulary
    /// under <see cref="T5Prefix"/>, and its canonical hyper-parameter and tokenizer keys.
    /// </summary>
    /// <param name="encoderFolder">The <c>text_encoder_2</c> folder (config.json and safetensors shards).</param>
    /// <param name="tokenizerFolder">The <c>tokenizer_2</c> folder (tokenizer.json).</param>
    /// <param name="weights">The canonical set the weights are added to.</param>
    /// <param name="resources">Receives the opened checkpoint.</param>
    /// <param name="prefix">The canonical weight prefix (FLUX: <see cref="T5Prefix"/>; SD3: <c>text_encoder_3.</c>).</param>
    /// <returns>The canonical <c>encoder_decoder.*</c> and <c>tokenizer.*</c> keys.</returns>
    public static IReadOnlyDictionary<string, object> AddT5Folder(string encoderFolder, string tokenizerFolder, CanonicalWeights weights, List<IDisposable> resources,
        string prefix = T5Prefix)
    {
        var config = HfConfig.Load(encoderFolder);
        var options = new HuggingFaceImportOptions();
        var mapping = new T5HfMapping();
        var checkpoint = SafetensorsCheckpoint.Open(encoderFolder);
        resources.Add(checkpoint);
        foreach (var (name, weight) in mapping.MapWeights(config, checkpoint.Tensors, options))
        {
            weights.Add(prefix + name, weight);
        }

        var metadata = new Dictionary<string, object>(mapping.MapHyperParameters(config, options), StringComparer.Ordinal);
        HfTokenizerTranslator.AddTo(metadata, tokenizerFolder, config);
        return metadata;
    }

    /// <summary>Copies T5 encoder keys: <c>encoder_decoder.*</c> to <c>flux.text_encoder_2.*</c>, <c>tokenizer.*</c> to <c>flux.tokenizer_2.tokenizer.*</c>.</summary>
    private static void AddT5Metadata(Dictionary<string, object> metadata, IReadOnlyDictionary<string, object> t5)
    {
        const string encoder = "encoder_decoder.";
        foreach (var (key, value) in t5)
        {
            if (key.StartsWith(encoder, StringComparison.Ordinal))
            {
                metadata[FluxConfig.Prefix + "text_encoder_2." + key[encoder.Length..]] = value;
            }
            else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
            {
                metadata[FluxConfig.Prefix + "tokenizer_2." + key] = value;
            }
        }
    }
}
