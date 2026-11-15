using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>hrm_text</c> (HRM-Text): a low stack (<c>model.L_module.layers.*</c>, canonical layers [0, S)) and a high stack
/// (<c>model.H_module.layers.*</c>, canonical layers [S, 2S)) run in hierarchical cycles (canonical <c>cycles.*</c>, the learned low-state init
/// <c>model.z_L_init</c>), weight-less RMS norms, a per-channel sigmoid attention gate and scaled embeddings. Checkpoints store the gate, q, k and v
/// projections fused (<c>attn.gqkv_proj</c>, in that order) and gate+up fused, or separately as transformers names them after loading.
/// <c>num_layers_per_stack</c> is S (a training-style config keeps it in <c>num_hidden_layers</c>, which transformers expands to the cache-slot
/// count). The prefix-LM attention over instruction tokens is not modelled: attention is causal.
/// </summary>
internal sealed partial class HrmTextHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("attn.o_proj.weight", "attention.o.weight"),
        ("self_attn.gate_proj.weight", "attention.gate.weight"));

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = With(LlamaGlobalTensors, ("model.z_L_init", "cycles.low_init"));

    [GeneratedRegex(@"^model\.([LH])_module\.layers\.(\d+)\.(.+)$")]
    private static partial Regex StackLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["hrm_text"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => StackLayerName();

    /// <summary>Group 1 is the stack, group 2 the layer within it, group 3 the suffix; the high stack follows the low one.</summary>
    protected override (string Layer, string Suffix) LayerTensorOf(HfConfig c, Match match)
    {
        int layer = int.Parse(match.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture) + (match.Groups[1].Value == "H" ? StackLayers(c) : 0);
        return (layer.ToString(System.Globalization.CultureInfo.InvariantCulture), match.Groups[3].Value);
    }

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => 2 * StackLayers(c);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.cycles.stack_layers"] = StackLayers(c);
        md["decoder.cycles.high"] = c.Int("H_cycles");
        md["decoder.cycles.low"] = c.Int("L_cycles");
        md["decoder.embedding_scale"] = (float)(c.GetDouble("embedding_scale") ?? 1.0 / (c.GetDouble("initializer_range") ?? 0.02));
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        int width = HeadCount(c) * HeadDimOf(c);
        switch (suffix)
        {
            case "attn.gqkv_proj.weight":
                string[] parts = ["attention.gate.weight", "attention.q.weight", "attention.k.weight", "attention.v.weight"];
                for (int i = 0; i < parts.Length; i++)
                {
                    collector.Add(HfTransforms.Plain(w.RowSlice("", i * width, width), layerPrefix + parts[i]));
                }

                return true;
            case "mlp.gate_up_proj.weight":
                int ffn = FeedForwardSize(c);
                collector.Add(HfTransforms.Plain(w.RowSlice("", 0, ffn), layerPrefix + "mlp.gate.weight"));
                collector.Add(HfTransforms.Plain(w.RowSlice("", ffn, ffn), layerPrefix + "mlp.up.weight"));
                return true;
            default:
                return false;
        }
    }

    private static int StackLayers(HfConfig c) => c.GetInt("num_layers_per_stack") ?? c.Int("num_hidden_layers");
}
