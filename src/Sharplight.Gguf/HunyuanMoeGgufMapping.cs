namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>hunyuan-moe</c> (Hunyuan-A13B): the Hunyuan dense layer (q/k norms after the rotary embedding, NTK-alpha rope base) with
/// softmax-scored, renormalized experts plus an always-on shared MLP.
/// </summary>
internal sealed class HunyuanMoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["hunyuan-moe"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        HunyuanDenseGgufMapping.AddHunyuanAttention(md, g, HeadDimOf(g));
        md[prefix + "moe.normalize_weights"] = true;
        md[prefix + "moe.router_activation"] = "softmax";
    }
}
