using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Assembles the canonical <see cref="ModelSource"/> of a latent-diffusion model (architecture <c>latent_diffusion</c>): the
/// canonical weights of the UNet, VAE and text encoder, their configurations, the tokenizer and the scheduler settings.
/// </summary>
internal static class LatentDiffusionSource
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "latent_diffusion";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>Canonical weight prefix of the (first) text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    /// <summary>The SD 1.x / 2.x latent-to-RGB preview factors ([4, 3], row-major).</summary>
    public static readonly float[] StableDiffusion1RgbFactors =
    [
        0.3512f, 0.2297f, 0.3227f,
        0.3250f, 0.4974f, 0.2350f,
        -0.2829f, 0.1762f, 0.2721f,
        -0.2120f, -0.2616f, -0.7177f,
    ];

    /// <summary>Builds the source.</summary>
    /// <param name="format">Informational container format.</param>
    /// <param name="name">Display name.</param>
    /// <param name="path">The file or folder opened.</param>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="unet">UNet configuration.</param>
    /// <param name="vae">VAE configuration.</param>
    /// <param name="text">Text encoder configuration.</param>
    /// <param name="vocabulary">The text encoder's tokenizer.</param>
    /// <param name="scheduler">Scheduler settings.</param>
    /// <param name="resources">The open files backing the weights.</param>
    public static ModelSource Create(string format, string name, string path, CanonicalWeights weights, UNetConfig unet, VaeConfig vae,
        ClipTextConfig text, ClipVocabulary vocabulary, SchedulerSettings scheduler, IReadOnlyList<IDisposable> resources)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = name,
            [CanonicalKeys.Architecture] = Architecture,
        };
        unet.WriteTo(metadata, Prefix);
        vae.WriteTo(metadata, Prefix);
        text.WriteTo(metadata, Prefix, "text_encoder");
        vocabulary.WriteTo(metadata, "", text.ContextLength);
        scheduler.WriteTo(metadata, Prefix);
        return new ModelSource
        {
            Format = format,
            Architecture = Architecture,
            Metadata = metadata,
            Weights = weights.Weights,
            OwnedResources = resources,
            Path = path,
        };
    }
}
