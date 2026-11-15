namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>cogvlm</c> (CogVLM's language model): Llama layers with fused <c>attn_qkv</c> and plain NeoX rope (base 10000), plus the visual
/// experts <c>vis_attn_qkv</c> / <c>vis_attn_output</c> / <c>vis_gate</c> / <c>vis_up</c> / <c>vis_down</c> as the layers' media weight set
/// (<c>layers.{i}.media.*</c>), which image rows use. The vision tower comes from the <c>cogvlm</c> projector file (<see cref="ClipGgufMapping"/>).
/// </summary>
internal sealed class CogVlmGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["vis_attn_output.weight"] = "media.attention.o.weight",
        ["vis_gate.weight"] = "media.mlp.gate.weight",
        ["vis_up.weight"] = "media.mlp.up.weight",
        ["vis_down.weight"] = "media.mlp.down.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["cogvlm"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_layer;
}
