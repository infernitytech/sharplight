namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>bitnet</c> (BitNet b1.58): the Llama layer with RMS sub-norms in front of the attention output projection
/// (<c>self_attn.attn_sub_norm</c>) and the MLP down projection (<c>mlp.ffn_sub_norm</c>) and a squared-ReLU gated MLP. Checkpoints
/// stored with unpacked (bf16) weights import directly; packed ternary checkpoints are not supported.
/// </summary>
internal sealed class BitnetHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.attn_sub_norm.weight", "attention.sub_norm.weight"),
        ("mlp.ffn_sub_norm.weight", "mlp.sub_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["bitnet"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;
}
