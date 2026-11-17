using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a FLUX.2 transformer with companion files loaded as auxiliary sources (its GGUF files hold the transformer only): the text
/// encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, a <c>decoder</c> language model, Qwen3-4B / -8B as a GGUF or the pipeline's
/// <c>text_encoder</c> folder, kept whole under <c>text_encoder.</c>) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, the FLUX.2 KL
/// autoencoder with its batch-norm statistics).
/// </summary>
public sealed class Flux2CompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = LanguageModel(Flux2Adapter.TextEncoderPrefix, "text_encoder_source", "decoder"),
        // The family's latent patching and batch-norm epsilon are kept: a stand-alone VAE file does not record them.
        [AuxiliaryRoles.Vae] = new([AutoencoderArchitecture], "vae.", name => name.StartsWith("vae.", StringComparison.Ordinal) ? name : null,
            AutoencoderArchitecture + ".vae.", "vae", null, ["scaling_factor", "shift_factor", "latent_rgb_factors", "latent_patch", "batch_norm_eps"]),
    };

    /// <inheritdoc/>
    public override string Name => "FLUX.2 with companion files";

    /// <inheritdoc/>
    protected override string Architecture => Flux2Adapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new Flux2Adapter().Build(new ModelSourceSet { Primary = merged }, options);
}
