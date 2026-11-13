using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Base of the Hugging Face mixture-of-experts decoders that map onto the canonical <c>decoder</c> MoE block (routing documented on
/// <c>MoeOptions</c>). It knows the router, selection-bias, shared-expert and expert tensor names these checkpoints use, stacks
/// per-expert tensors into rank-3 weights, drops the multi-token-prediction layers that follow the decoder layers, and writes the
/// canonical <c>moe.*</c> keys from a <see cref="MoeRouting"/> description.
/// </summary>
internal abstract partial class HfMoeFamilyMapping : HfDecoderFamilyMapping
{
    /// <summary>MoE tensors that are only renamed (selection biases become F32 [E]), HF layer suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_moeTensors = new(StringComparer.Ordinal)
    {
        ["mlp.gate.weight"] = "mlp.router.weight",
        ["mlp.gate.wg.weight"] = "mlp.router.weight",
        ["mlp.router.gate.weight"] = "mlp.router.weight",
        ["block_sparse_moe.gate.weight"] = "mlp.router.weight",
        ["mlp.gate.e_score_correction_bias"] = "mlp.router_bias",
        ["mlp.gate.expert_bias"] = "mlp.router_bias",
        ["mlp.expert_bias"] = "mlp.router_bias",
        ["mlp.e_score_correction_bias"] = "mlp.router_bias",
        ["mlp.moe_statics.e_score_correction_bias"] = "mlp.router_bias",
        ["block_sparse_moe.e_score_correction_bias"] = "mlp.router_bias",
        ["mlp.experts.gate_up_proj"] = "mlp.experts.gate_up.weight",
        ["mlp.experts.down_proj"] = "mlp.experts.down.weight",
        ["mlp.shared_experts.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_experts.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_experts.down_proj.weight"] = "mlp.shared.down.weight",
        ["mlp.shared_expert.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_expert.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_expert.down_proj.weight"] = "mlp.shared.down.weight",
        ["mlp.shared_mlp.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_mlp.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_mlp.down_proj.weight"] = "mlp.shared.down.weight",
    };

    /// <summary>Per-expert projection names (Llama-style and Mixtral-style w1/w3/w2) to the canonical expert kind.</summary>
    private static readonly Dictionary<string, string> s_expertKinds = new(StringComparer.Ordinal)
    {
        ["gate_proj"] = "gate",
        ["up_proj"] = "up",
        ["down_proj"] = "down",
        ["w1"] = "gate",
        ["w3"] = "up",
        ["w2"] = "down",
    };

    // Per-expert tensors of one import, keyed by that import's collector (the mapping itself is shared).
    private static readonly ConditionalWeakTable<HfWeightCollector, HfPartJoiner> s_experts = [];

    [GeneratedRegex(@"^(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(\w+)\.weight$")]
    private static partial Regex PerExpertName();

    [GeneratedRegex(@"^layers\.(\d+)\.$")]
    private static partial Regex LayerPrefix();

    /// <summary>The routing of this checkpoint, or null for a checkpoint without experts.</summary>
    protected abstract MoeRouting? Routing(HfConfig c);

    /// <summary>Maps a family-specific layer tensor (fused attention, renamed norms); false when it is not one.</summary>
    protected virtual bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector) => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        AddFamilyHyperParameters(md, c);
        if (Routing(c) is { } routing)
        {
            routing.AddTo(md, "decoder.");
        }
    }

    /// <summary>Family-specific hyper-parameters besides the routing.</summary>
    protected virtual void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
    }

    /// <inheritdoc/>
    protected sealed override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (int.Parse(LayerPrefix().Match(layerPrefix).Groups[1].Value) >= LayerCount(c))
        {
            return true;   // multi-token-prediction layers after the decoder are not used
        }

        if (TryMapFamilyTensor(c, layerPrefix, suffix, w, collector))
        {
            return true;
        }

        if (s_moeTensors.TryGetValue(suffix, out var canonical))
        {
            collector.Add(canonical == "mlp.router_bias"
                ? HfTransforms.ToF32(w, layerPrefix + canonical, new Shape(w.Shape.Elements()))
                : HfTransforms.Plain(w, layerPrefix + canonical));
            return true;
        }

        if (PerExpertName().Match(suffix) is { Success: true } match && s_expertKinds.TryGetValue(match.Groups[2].Value, out var kind))
        {
            s_experts.GetValue(collector, _ => HfPartJoiner.Experts()).Add($"{layerPrefix}mlp.experts.{kind}.weight", int.Parse(match.Groups[1].Value), w);
            return true;
        }

        return false;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        if (s_experts.TryGetValue(collector, out var experts))
        {
            experts.JoinInto(collector, Routing(c)?.Experts ?? 0);
            s_experts.Remove(collector);
        }
    }
}
