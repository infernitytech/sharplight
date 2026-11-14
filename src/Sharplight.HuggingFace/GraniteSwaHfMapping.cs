using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>granite_swa</c> / <c>granitemoe_swa</c> / <c>granitemoeshared</c> (Granite with sliding-window layers): Granite's muP multipliers, per-head
/// attention sinks, sliding layers from <c>layer_types</c> and a rotary base per layer (<c>layer_rope_theta</c>, 0 = NoPE layer; or the older
/// <c>no_rope_layers</c> flags). The MoE variants route with softmax over the top-k logits (fused <c>experts.gate_up_proj</c>, gate rows first) and add a
/// shared MLP whose fused <c>input_linear</c> holds the gate rows, then the up rows.
/// </summary>
/// <remarks>
/// The canonical decoder has one rotary base for the sliding layers and one for the others, so every sliding layer with rotary must share a theta, and so
/// must every other layer with rotary.
/// </remarks>
internal sealed class GraniteSwaHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.sinks", "attention.sinks"),
        ("block_sparse_moe.router.weight", "mlp.router.weight"),
        ("block_sparse_moe.router.layer.weight", "mlp.router.weight"),
        ("block_sparse_moe.experts.gate_up_proj", "mlp.experts.gate_up.weight"),
        ("block_sparse_moe.experts.down_proj", "mlp.experts.down.weight"),
        ("shared_mlp.output_linear.weight", "mlp.shared.down.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["granite_swa", "granitemoe_swa", "granitemoeshared"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        GraniteHfMapping.AddMupScales(md, c);
        var sliding = AddSlidingLayers(md, c) ?? new int[LayerCount(c)];
        AddRotaryLayers(md, c, sliding);
        if (c.GetInt("num_local_experts") is > 0 and var experts)
        {
            md[prefix + "moe.expert_count"] = experts;
            md[prefix + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
            md[prefix + "moe.ffn_hidden_size"] = c.Int("intermediate_size");
            md[prefix + "moe.normalize_weights"] = true;
        }
    }

    private void AddRotaryLayers(Dictionary<string, object> md, HfConfig c, int[] sliding)
    {
        const string prefix = "decoder.";
        if (IntArray(c, "no_rope_layers") is { } ropeFlags)
        {
            md[prefix + "rope.layers"] = ropeFlags;
            return;
        }

        if (c.Get("layer_rope_theta") is not { ValueKind: JsonValueKind.Array } thetaArray)
        {
            return;
        }

        var thetas = thetaArray.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        md[prefix + "rope.layers"] = thetas.Select(theta => theta != 0 ? 1 : 0).ToArray();
        if (UniformTheta(thetas, sliding, slidingLayers: false) is { } globalBase)
        {
            md[prefix + "rope.base"] = globalBase;
        }

        if (UniformTheta(thetas, sliding, slidingLayers: true) is { } localBase)
        {
            md[prefix + "rope.local_base"] = localBase;
        }
    }

    /// <summary>The theta all rotary layers of one kind share (null when no such layer has rotary).</summary>
    /// <exception cref="NotSupportedException">Layers of the kind use different thetas.</exception>
    private static float? UniformTheta(double[] thetas, int[] sliding, bool slidingLayers)
    {
        var distinct = thetas.Where((theta, layer) => theta != 0 && (sliding[layer] != 0) == slidingLayers).Distinct().ToArray();
        return distinct.Length switch
        {
            0 => null,
            1 => (float)distinct[0],
            _ => throw new NotSupportedException($"granite_swa: {(slidingLayers ? "sliding" : "full")} attention layers with different rope thetas are not supported."),
        };
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != "shared_mlp.input_linear.weight")
        {
            return false;
        }

        long shared = w.Shape[0].Fixed / 2;
        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, shared), layerPrefix + "mlp.shared.gate.weight"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", shared, shared), layerPrefix + "mlp.shared.up.weight"));
        return true;
    }
}
