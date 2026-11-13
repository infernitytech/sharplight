using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>nemotron_h</c> / <c>nemotron_h_moe</c> (Nemotron-H / Nemotron-3 Nano hybrid Mamba-2 + attention + MLP or mixture of experts) → canonical <c>nemotron_h</c> (documented on <c>NemotronHAdapter</c>).</summary>
internal sealed class NemotronHGgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["nemotron_h", "nemotron_h_moe"];
    /// <inheritdoc/>
    public string Architecture => "nemotron_h";

    /// <summary>A per-layer GGUF array (or a scalar repeated, default 0) as ints.</summary>
    private static int[] PerLayer(GgufMetadataView g, string key, int layerCount)
        => g.Raw(key) is Array values && values.Length == layerCount
            ? Enumerable.Range(0, layerCount).Select(i => Convert.ToInt32(values.GetValue(i))).ToArray()
            : Enumerable.Repeat(g.Int(key, 0), layerCount).ToArray();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        var kvHeadCounts = PerLayer(g, "attention.head_count_kv", layerCount);
        var ffnLengths = PerLayer(g, "feed_forward_length", layerCount);
        bool moeArchitecture = g.GgufArchitecture == "nemotron_h_moe";   // there a layer with a feed-forward size is a mixture-of-experts layer
        var kinds = Enumerable.Range(0, layerCount)
            .Select(il => kvHeadCounts[il] > 0 ? "attention" : ffnLengths[il] > 0 ? (moeArchitecture ? "moe" : "mlp") : "ssm")
            .ToArray();
        int ssmHeads = g.Int("ssm.time_step_rank");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "block_kinds"] = kinds,
            [a + "mlp.activation"] = "relu2",
            [a + "ssm.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "ssm.state_size"] = g.Int("ssm.state_size"),
            [a + "ssm.group_count"] = g.Int("ssm.group_count"),
            [a + "ssm.head_count"] = ssmHeads,
            [a + "ssm.head_dim"] = g.Int("ssm.inner_size") / ssmHeads,
        };
        if (kvHeadCounts.Any(k => k > 0))
        {
            md[a + "attention.head_count"] = g.Int("attention.head_count");
            md[a + "attention.kv_head_count"] = kvHeadCounts.Max();
            md[a + "attention.head_dim"] = g.Int("attention.key_length", g.Int("embedding_length") / g.Int("attention.head_count"));
        }

        GgufMappingHelpers.AddContextLength(md, a, g);

        if (moeArchitecture)
        {
            if (g.Int("expert_group_count", 1) > 1)
            {
                throw new NotSupportedException("nemotron_h_moe: group-limited expert routing is not supported yet.");
            }

            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.normalize_weights"] = g.Bool("expert_weights_norm", true);
            if (g.Has("expert_weights_scale") && g.Float("expert_weights_scale") is var weightsScale && weightsScale != 1f)
            {
                md[a + "moe.weights_scale"] = weightsScale;
            }
        }

        return md;
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ssm_in.weight"] = "ssm.in.weight",
        ["ssm_out.weight"] = "ssm.out.weight",
        ["ssm_conv1d.weight"] = "ssm.conv1d.weight",
        ["ssm_conv1d.bias"] = "ssm.conv1d.bias",
        ["ssm_dt.bias"] = "ssm.dt_bias",
        ["ssm_a"] = "ssm.a",
        ["ssm_d"] = "ssm.d",
        ["ssm_norm.weight"] = "ssm.norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "moe.router.weight",
        ["exp_probs_b.bias"] = "moe.router_bias",
        ["ffn_up_exps.weight"] = "moe.experts.up.weight",
        ["ffn_down_exps.weight"] = "moe.experts.down.weight",
        ["ffn_up_shexp.weight"] = "moe.shared.up.weight",
        ["ffn_down_shexp.weight"] = "moe.shared.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>A, D are stored [H, 1] and the gated norm [G, inner/G]; canonical weights are flat.</summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (w.Name.EndsWith(".ssm.a", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.d", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.norm.weight", StringComparison.Ordinal))
        {
            return w.As(w.Name, new Shape(w.Shape.Elements()));
        }

        return w;
    }
}
