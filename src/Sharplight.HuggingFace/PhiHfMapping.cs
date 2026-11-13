namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>phi</c> (Phi-1/1.5/2): layer norm with bias, attention and MLP in parallel around one shared norm, partial rotary, plain biased GELU MLP
/// (<c>fc1</c>/<c>fc2</c>), output projection <c>dense</c>, biased output head. Per-head q/k layer norms (<c>qk_layernorm</c>) are not supported.
/// </summary>
internal sealed class PhiHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.dense.weight", "attention.o.weight"),
        ("self_attn.dense.bias", "attention.o.bias"),
        ("mlp.fc1.weight", "mlp.up.weight"),
        ("mlp.fc1.bias", "mlp.up.bias"),
        ("mlp.fc2.weight", "mlp.down.weight"),
        ("mlp.fc2.bias", "mlp.down.bias"));

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = With(LlamaGlobalTensors,
        ("model.final_layernorm.weight", "output_norm.weight"),
        ("model.final_layernorm.bias", "output_norm.bias"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["phi"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("qk_layernorm") == true)
        {
            throw new NotSupportedException("phi: per-head q/k layer norms (qk_layernorm) are not supported.");
        }

        md["decoder.residual.parallel"] = true;
    }
}
