namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>phimoe</c> (Phi-3.5-MoE): biased layer norms, biased attention with an optional sliding window and LongRoPE factors, top-2
/// SwiGLU experts and a biased output head. The file does not state the routing; it is mapped to the model's own sparsemixer routing
/// (jitter ε = 0.01, Phi-3.5-MoE's <c>router_jitter_noise</c>) and mean-centred layer norms, as in transformers. llama.cpp instead runs the
/// architecture through its phi3 graph with RMS norms (plus the bias) and softmax top-2 routing with renormalized weights, so its output
/// differs from both.
/// </summary>
internal sealed class PhimoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["phimoe"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "moe.router_activation"] = "sparsemixer";
        md[GgufDecoderHyperParameters.Prefix + "moe.sparsemixer_jitter"] = 0.01f;
        if (g.Int("attention.sliding_window", 0) is > 0 and var window)
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.sliding_window"] = window;   // every layer; absent / 0 = full attention
        }
    }
}
