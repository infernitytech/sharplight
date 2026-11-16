using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes an SD3 transformer with companion files loaded as auxiliary sources (the SD3 / SD3.5 GGUF files hold the transformer
/// only): CLIP-L (role <see cref="AuxiliaryRoles.TextEncoder"/>), OpenCLIP bigG (role <see cref="AuxiliaryRoles.SecondTextEncoder"/>),
/// both <c>clip_text_encoder</c> sources, the T5 encoder (role <see cref="AuxiliaryRoles.ThirdTextEncoder"/>, any <c>encoder_decoder</c>
/// source) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>), renamed into the <c>sd3</c> vocabulary of <see cref="Sd3Adapter"/>. SD3
/// conditions on the CLIP encoders' penultimate layers without the final norm and pads CLIP-G prompts with <c>!</c>: those keys stay the
/// primary's.
/// </summary>
public sealed class Sd3CompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = Clip("text_encoder.", "text_encoder", "", "clip_skip", "final_norm"),
        [AuxiliaryRoles.SecondTextEncoder] = Clip("text_encoder_2.", "text_encoder_2", null, "clip_skip", "final_norm", "pad_id"),
        [AuxiliaryRoles.ThirdTextEncoder] = T5("text_encoder_3.", "text_encoder_3", "tokenizer_3."),
        [AuxiliaryRoles.Vae] = Vae(),
    };

    /// <inheritdoc/>
    public override string Name => "Stable Diffusion 3 with companion files";

    /// <inheritdoc/>
    protected override string Architecture => Sd3Adapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new Sd3Adapter().Build(new ModelSourceSet { Primary = merged }, options);
}
