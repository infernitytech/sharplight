using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>nemotron</c> (Nemotron-4, Minitron, Nemotron-Mini): layer norm with bias in the <c>1 + w</c> convention (folded on import), squared-ReLU MLP without gate
/// (<c>up_proj</c>/<c>down_proj</c>), partial rotary.
/// </summary>
internal sealed class NemotronDenseHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["nemotron"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override WeightInfo NormWeight(WeightInfo w, string name) => HfTransforms.NormPlusOne(w, name);
}
