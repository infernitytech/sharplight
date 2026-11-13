using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>gemma</c> (Gemma 1): multi-query attention, GeGLU with the tanh GELU, embeddings scaled by sqrt(hidden), RMS norms in the <c>1 + w</c> convention
/// (folded on import); no sandwich norms (<c>post_attention_layernorm</c> is the norm in front of the MLP).
/// </summary>
internal sealed class Gemma1HfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["gemma"];

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "gelu_tanh";

    /// <inheritdoc/>
    protected override WeightInfo NormWeight(WeightInfo w, string name) => HfTransforms.NormPlusOne(w, name);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.embedding_scale"] = MathF.Sqrt(HiddenSize(c));
    }
}
