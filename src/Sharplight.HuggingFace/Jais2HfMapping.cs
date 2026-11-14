namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>jais2</c> (Jais 2): the Llama layout with layer norms (with bias), neox rotary and a plain squared-ReLU MLP, optionally biased.</summary>
internal sealed class Jais2HfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["jais2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;
}
