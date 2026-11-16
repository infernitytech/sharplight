using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a Qwen-Image transformer with companion files loaded as auxiliary sources (the Qwen-Image GGUF files hold the transformer
/// only): the text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, a <c>decoder</c> language model such as Qwen2.5-VL-7B as a GGUF
/// or the pipeline's <c>text_encoder</c> folder, kept whole under <c>text_encoder.</c>) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, a
/// stand-alone causal 3-D VAE file, architecture <c>wan_autoencoder</c>).
/// </summary>
public sealed class QwenImageCompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = LanguageModel(QwenImageAdapter.TextEncoderPrefix, "text_encoder_source", "decoder"),
        [AuxiliaryRoles.Vae] = CausalVae(),
    };

    /// <inheritdoc/>
    public override string Name => "Qwen-Image with companion files";

    /// <inheritdoc/>
    protected override string Architecture => QwenImageAdapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new QwenImageAdapter().Build(new ModelSourceSet { Primary = merged }, options);
}
