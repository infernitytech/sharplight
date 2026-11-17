using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a Krea 2 transformer with companion files loaded as auxiliary sources (its original-name, ComfyUI and GGUF files hold the
/// transformer only): the text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, a <c>decoder</c> language model, Qwen3-VL-4B as a GGUF or
/// the pipeline's <c>text_encoder</c> folder, kept whole under <c>text_encoder.</c>) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, the
/// Qwen-Image VAE: ComfyUI's <c>qwen_image_vae.safetensors</c> or the pipeline's file, architecture <c>wan_autoencoder</c>).
/// </summary>
public sealed class Krea2CompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = LanguageModel(Krea2Adapter.TextEncoderPrefix, "text_encoder_source", "decoder"),
        [AuxiliaryRoles.Vae] = CausalVae(),
    };

    /// <inheritdoc/>
    public override string Name => "Krea 2 with companion files";

    /// <inheritdoc/>
    protected override string Architecture => Krea2Adapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new Krea2Adapter().Build(new ModelSourceSet { Primary = merged }, options);
}
