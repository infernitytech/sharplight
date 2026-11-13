namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>minicpm</c> (MiniCPM 2B, MiniCPM4/4.1): the Llama decoder with muP multipliers — embeddings scaled by <c>scale_emb</c>, every sub-layer output by
/// <c>scale_depth / sqrt(layers)</c>, logits by <c>dim_model_base / hidden_size</c>; longrope scaling.
/// </summary>
internal sealed class MiniCpmHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["minicpm"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        md[prefix + "embedding_scale"] = (float)(c.GetDouble("scale_emb") ?? 1.0);
        md[prefix + "residual.scale"] = (float)((c.GetDouble("scale_depth") ?? 1.0) / Math.Sqrt(LayerCount(c)));
        md[prefix + "logit_scale"] = (float)((c.GetDouble("dim_model_base") ?? c.Int("hidden_size")) / c.Int("hidden_size"));
    }
}
