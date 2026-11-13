using System.Runtime.CompilerServices;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>gpt_oss</c> (gpt-oss-20b / 120b): biased q/k/v/o projections, per-head attention sinks, sliding-window layers from
/// <c>layer_types</c> (every layer uses the YaRN-scaled rotary embedding), and a MoE block with a biased router, softmax over the top-k
/// logits, per-expert biases and the clamped SwiGLU of gpt-oss (<c>swiglu_limit</c>). The fused experts are stored [E, in, out] with
/// gate and up columns interleaved (<c>gate_up_proj</c>) — as BF16 or as MXFP4 <c>*_blocks</c> / <c>*_scales</c> pairs — and become
/// the canonical [E, out, in] <c>experts.gate_up</c> (gate rows first) and <c>experts.down</c> weights.
/// </summary>
internal sealed class GptOssHfMapping : HfMoeFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.sinks", "attention.sinks"),
        ("mlp.router.weight", "mlp.router.weight"));

    // MXFP4 blocks waiting for their scales, per import (keyed by the import's collector).
    private static readonly ConditionalWeakTable<HfWeightCollector, Dictionary<string, WeightInfo>> s_blocks = [];

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["gpt_oss"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "swiglu_oai";

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts"), c.Int("num_experts_per_tok", "experts_per_token"), c.Int("intermediate_size"), "softmax_topk", Normalize: false);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.mlp.swiglu_limit"] = (float)(c.GetDouble("swiglu_limit") ?? 7.0);
        if (AddSlidingLayers(md, c) is not null)
        {
            md["decoder.rope.local_freq_factors"] = true;
        }
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">MXFP4 scales arrive without their blocks.</exception>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string prefix = layerPrefix + "mlp.experts.";
        switch (suffix)
        {
            case "mlp.router.bias":
                collector.Add(HfTransforms.Plain(w, layerPrefix + "mlp.router.bias"));
                return true;
            case "mlp.experts.gate_up_proj":
                collector.Add(HfExpertLayouts.Transposed(w, prefix + "gate_up.weight", HfExpertLayouts.DeinterleavedPairs((int)w.Shape[2].Fixed)));
                return true;
            case "mlp.experts.gate_up_proj_bias":
                collector.Add(HfExpertLayouts.GatherColumns(w, prefix + "gate_up.bias", HfExpertLayouts.DeinterleavedPairs((int)w.Shape[1].Fixed)));
                return true;
            case "mlp.experts.down_proj":
                collector.Add(HfExpertLayouts.Transposed(w, prefix + "down.weight"));
                return true;
            case "mlp.experts.down_proj_bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "down.bias"));
                return true;
            case "mlp.experts.gate_up_proj_blocks" or "mlp.experts.down_proj_blocks":
                s_blocks.GetOrCreateValue(collector)[layerPrefix + suffix] = w;
                return true;
            case "mlp.experts.gate_up_proj_scales" or "mlp.experts.down_proj_scales":
                string blocksName = layerPrefix + suffix.Replace("_scales", "_blocks", StringComparison.Ordinal);
                if (!s_blocks.GetOrCreateValue(collector).Remove(blocksName, out var blocks))
                {
                    throw new InvalidDataException($"gpt_oss: {layerPrefix}{suffix} has no matching blocks tensor.");
                }

                bool gateUp = suffix.StartsWith("mlp.experts.gate_up", StringComparison.Ordinal);
                collector.Add(HfExpertLayouts.Mxfp4(blocks, w, prefix + (gateUp ? "gate_up.weight" : "down.weight"),
                    gateUp ? HfExpertLayouts.DeinterleavedPairs((int)w.Shape[1].Fixed) : null));
                return true;
            default:
                return false;
        }
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        base.FinishWeights(c, collector);
        if (s_blocks.TryGetValue(collector, out var pending))
        {
            foreach (var name in pending.Keys)
            {
                collector.AddUnmapped(name);
            }

            s_blocks.Remove(collector);
        }
    }
}
