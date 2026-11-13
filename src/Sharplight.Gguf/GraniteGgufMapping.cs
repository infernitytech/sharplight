namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>granite</c> and <c>granitemoe</c> (IBM Granite 3.x dense and fused-expert MoE): the llama graph with muP multipliers
/// (embedding, residual, attention and logit scale). Rope is NoPE when <c>rope.scaling.finetuned</c> is false (Granite 4 style files).
/// </summary>
internal sealed class GraniteGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["granite", "granitemoe"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        GgufMupScales.Add(md, g);
        if (!g.Bool("rope.scaling.finetuned", true))
        {
            md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = new int[g.Int("block_count")];
        }
    }
}
