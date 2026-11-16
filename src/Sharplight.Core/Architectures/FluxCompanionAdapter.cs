using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a FLUX transformer with companion files loaded as auxiliary sources (transformer-only GGUF / single files are
/// distributed without their text encoders and VAE): the CLIP-L text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>,
/// architecture <c>clip_text_encoder</c>), the T5 encoder (role <see cref="AuxiliaryRoles.SecondTextEncoder"/>, any
/// <c>encoder_decoder</c> source such as a T5 encoder GGUF) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, architecture
/// <c>kl_autoencoder</c>), renamed into the <c>flux</c> vocabulary of <see cref="FluxAdapter"/>.
/// </summary>
public sealed class FluxCompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = Clip("text_encoder.", "text_encoder", ""),
        [AuxiliaryRoles.SecondTextEncoder] = T5("text_encoder_2.", "text_encoder_2", "tokenizer_2."),
        [AuxiliaryRoles.Vae] = Vae(),
    };

    /// <inheritdoc/>
    public override string Name => "FLUX with companion files";

    /// <inheritdoc/>
    protected override string Architecture => FluxAdapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new FluxAdapter().Build(new ModelSourceSet { Primary = merged }, options);
}
