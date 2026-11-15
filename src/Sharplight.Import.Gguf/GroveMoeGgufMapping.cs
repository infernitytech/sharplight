namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>grovemoe</c> (GroveMoE): Qwen3-MoE layers (per-head q/k norms, softmax experts renormalized over the selection) plus adjugate
/// experts <c>ffn_{gate,up,down}_chexps</c>, one per <c>experts_per_group</c> consecutive routed experts, whose output is scaled by
/// <c>expert_group_scale</c>. The canonical block follows the reference implementation: each slot runs its group's adjugate expert on the
/// normed layer input with the slot's routing weight. llama.cpp 8ca5a80's graph instead feeds the adjugate experts the routed experts' output
/// and weights them by the probability of expert id / group size, so its output differs.
/// </summary>
internal sealed class GroveMoeGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["ffn_gate_chexps.weight"] = "mlp.adjugate_experts.gate.weight",
        ["ffn_up_chexps.weight"] = "mlp.adjugate_experts.up.weight",
        ["ffn_down_chexps.weight"] = "mlp.adjugate_experts.down.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["grovemoe"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "moe.normalize_weights"] = true;
        md[prefix + "moe.adjugate_group_size"] = g.Int("experts_per_group");
        md[prefix + "moe.adjugate_scale"] = g.Float("expert_group_scale");
    }
}
