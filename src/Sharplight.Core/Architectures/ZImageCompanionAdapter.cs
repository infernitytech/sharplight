using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a Z-Image transformer with companion files loaded as auxiliary sources (the Z-Image GGUF files hold the transformer only):
/// the text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, a <c>decoder</c> language model such as Qwen3-4B as a GGUF or a
/// Hugging Face folder, kept whole under <c>text_encoder.</c>) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, FLUX's).
/// </summary>
public sealed class ZImageCompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = LanguageModel(ZImageAdapter.TextEncoderPrefix, "text_encoder_source", "decoder"),
        [AuxiliaryRoles.Vae] = Vae(),
    };

    /// <inheritdoc/>
    public override string Name => "Z-Image with companion files";

    /// <inheritdoc/>
    protected override string Architecture => ZImageAdapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new ZImageAdapter().Build(new ModelSourceSet { Primary = merged }, options);
}
