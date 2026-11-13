using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>mixtral</c> (Mixtral 8x7B/8x22B): the Mistral decoder with a softmax top-k router over experts whose weights are renormalized.
/// Checkpoints store the experts either per expert (<c>block_sparse_moe.experts.N.w1/w2/w3</c>, w1 = gate, w3 = up, w2 = down) or fused
/// per layer (<c>mlp.experts.gate_up_proj</c> [E, 2·ff, H] with the gate first, <c>mlp.experts.down_proj</c>); both become the canonical
/// rank-3 expert weights.
/// </summary>
internal sealed partial class MixtralHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("mlp.gate_proj.weight", null), ("mlp.up_proj.weight", null), ("mlp.down_proj.weight", null),
        ("block_sparse_moe.gate.weight", "mlp.router.weight"),
        ("mlp.gate.weight", "mlp.router.weight"),
        ("mlp.experts.down_proj", "mlp.experts.down.weight"));

    // Per-expert tensors collected during one import, keyed by that import's collector (the mapping itself is shared).
    private static readonly ConditionalWeakTable<HfWeightCollector, SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>>> s_experts = [];

    [GeneratedRegex(@"^block_sparse_moe\.experts\.(\d+)\.w([123])\.weight$")]
    private static partial Regex PerExpertName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["mixtral"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        md[prefix + "moe.expert_count"] = c.Int("num_local_experts", "num_experts");
        md[prefix + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        md[prefix + "moe.ffn_hidden_size"] = c.Int("intermediate_size");
        md[prefix + "moe.normalize_weights"] = true;
        if (c.GetInt("sliding_window") is > 0 and var window)
        {
            md[prefix + "attention.sliding_window"] = window;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        int layer = int.Parse(layerPrefix.Split('.')[1]);
        if (PerExpertName().Match(suffix) is { Success: true } match)
        {
            string kind = match.Groups[2].Value switch { "1" => "gate", "2" => "down", _ => "up" };
            var experts = s_experts.GetOrCreateValue(collector);
            if (!experts.TryGetValue((layer, kind), out var byIndex))
            {
                experts[(layer, kind)] = byIndex = [];
            }

            byIndex[int.Parse(match.Groups[1].Value)] = w;
            return true;
        }

        if (suffix == "mlp.experts.gate_up_proj")
        {
            int ffn = (int)(w.Shape[1].Fixed / 2);
            var (gate, up) = HfFusedProjections.SplitSecondAxis(w, layerPrefix + "mlp.experts.gate.weight", layerPrefix + "mlp.experts.up.weight", ffn);
            collector.Add(gate);
            collector.Add(up);
            return true;
        }

        return false;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        int expertCount = c.Int("num_local_experts", "num_experts");
        foreach (var ((layer, kind), byIndex) in s_experts.GetOrCreateValue(collector))
        {
            if (byIndex.Count != expertCount || byIndex.Keys.Max() != expertCount - 1)
            {
                throw new InvalidDataException($"layer {layer}: expected {expertCount} {kind} experts, found {byIndex.Count}.");
            }

            collector.Add(WeightInfo.Stack($"layers.{layer}.mlp.experts.{kind}.weight", [.. byIndex.Values]));
        }

        s_experts.Remove(collector);
    }
}
