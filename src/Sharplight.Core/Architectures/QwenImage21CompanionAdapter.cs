using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a Qwen-Image 2.1 transformer with companion files loaded as auxiliary sources (its GGUF and ComfyUI files hold the transformer
/// only): the text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, a <c>decoder</c> language model, Qwen3-VL-8B as a GGUF or the
/// pipeline's <c>text_encoder</c> folder, kept whole under <c>text_encoder.</c>) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, the
/// pipeline's residual VAE file, architecture <c>wan_autoencoder</c>).
/// </summary>
public sealed class QwenImage21CompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = LanguageModel(QwenImage21Adapter.TextEncoderPrefix, "text_encoder_source", "decoder"),
        [AuxiliaryRoles.Vae] = CausalVae(),
    };

    /// <inheritdoc/>
    public override string Name => "Qwen-Image with companion files";

    /// <inheritdoc/>
    protected override string Architecture => QwenImage21Adapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new QwenImage21Adapter().Build(new ModelSourceSet { Primary = merged }, options);
}
