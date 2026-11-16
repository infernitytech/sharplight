using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a Chroma transformer with companion files loaded as auxiliary sources (the Chroma GGUF files hold the transformer only):
/// the T5 encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>, any <c>encoder_decoder</c> source such as a T5 encoder GGUF; Chroma
/// has no CLIP encoder) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, FLUX's), renamed into the <c>chroma</c> vocabulary of
/// <see cref="ChromaAdapter"/>.
/// </summary>
public sealed class ChromaCompanionAdapter : DiffusionCompanionAdapter
{
    private static readonly Dictionary<string, CompanionSlot> s_slots = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TextEncoder] = T5("text_encoder.", "text_encoder", ""),
        [AuxiliaryRoles.Vae] = Vae(),
    };

    /// <inheritdoc/>
    public override string Name => "Chroma with companion files";

    /// <inheritdoc/>
    protected override string Architecture => ChromaAdapter.Architecture;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, CompanionSlot> Slots => s_slots;

    /// <inheritdoc/>
    protected override ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options)
        => new ChromaAdapter().Build(new ModelSourceSet { Primary = merged }, options);
}
