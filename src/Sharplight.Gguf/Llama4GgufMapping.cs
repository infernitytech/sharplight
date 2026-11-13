namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>llama4</c> (Llama 4 Scout / Maverick text): interleaved rotary layers (every fourth layer is a NoPE layer with global
/// attention and position-dependent query temperature), chunked attention (8192) and an unweighted q/k norm on the rotary layers,
/// q/k rows stored permuted for llama.cpp's interleaved rope, and MoE layers (every <c>interleave_moe_layer_step</c>-th) with sigmoid
/// gates applied to the expert inputs plus a shared expert.
/// </summary>
internal sealed class Llama4GgufMapping : GgufDecoderFamilyMapping
{
    private const int NoRopeLayerStep = 4;
    private const int ChunkSize = 8192;
    private const float TemperatureScale = 0.1f;
    private const float TemperatureLength = 8192f;
    private const float TemperatureOffset = 1f;
    private const int ExpertsWithoutQkNorm = 128;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["llama4"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "moe.router_activation"] = "sigmoid";
        md[prefix + "moe.weights_before_experts"] = true;

        // A file with sliding_window 0 is a MobileLLM-style model: rotary everywhere, no chunking, no temperature.
        if (g.Has("attention.sliding_window") && g.Int("attention.sliding_window") == 0)
        {
            return;
        }

        int layerCount = TrunkLayerCount(g);
        var rope = Enumerable.Range(0, layerCount).Select(il => (il + 1) % NoRopeLayerStep != 0 ? 1 : 0).ToArray();
        var noRope = rope.Select(flag => 1 - flag).ToArray();
        md[prefix + "rope.layers"] = rope;
        md[prefix + "attention.chunk_size"] = ChunkSize;
        md[prefix + "attention.chunk_layers"] = rope;
        md[prefix + "attention.temperature_scale"] = TemperatureScale;
        md[prefix + "attention.temperature_length"] = TemperatureLength;
        md[prefix + "attention.temperature_offset"] = TemperatureOffset;
        md[prefix + "attention.temperature_layers"] = noRope;
        if (g.Int("expert_count", 0) != ExpertsWithoutQkNorm)
        {
            md[prefix + "attention.qk_norm_unweighted"] = true;
            md[prefix + "attention.qk_norm_layers"] = rope;
            md[prefix + "attention.qk_norm_after_rope"] = true;
        }
    }
}
