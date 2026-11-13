using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>nemotron_h</c> / <c>nemotron_h_moe</c> (Nemotron-H / Nemotron-3 Nano hybrid Mamba-2 + attention + MLP or mixture of experts) → canonical <c>nemotron_h</c> (documented on <c>NemotronHAdapter</c>).</summary>
public sealed partial class NemotronHGgufMapping : IGgufArchitectureMapping
{
    public IReadOnlyList<string> GgufArchitectures { get; } = ["nemotron_h", "nemotron_h_moe"];
    public string Architecture => "nemotron_h";

    private static int[] PerLayer(GgufMetadataView g, string key, int nLayer)
        => g.Raw(key) is Array arr && arr.Length == nLayer
            ? Enumerable.Range(0, nLayer).Select(i => Convert.ToInt32(arr.GetValue(i))).ToArray()
            : Enumerable.Repeat(g.Int(key, 0), nLayer).ToArray();

    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int nLayer = g.Int("block_count");
        var kv = PerLayer(g, "attention.head_count_kv", nLayer);
        var ffn = PerLayer(g, "feed_forward_length", nLayer);
        bool moeArch = g.GgufArchitecture == "nemotron_h_moe";   // there a layer with a feed-forward size is a mixture-of-experts layer
        var kinds = Enumerable.Range(0, nLayer).Select(il => kv[il] > 0 ? "attention" : ffn[il] > 0 ? (moeArch ? "moe" : "mlp") : "ssm").ToArray();
        int heads = g.Int("ssm.time_step_rank");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "block_kinds"] = kinds,
            [a + "mlp.activation"] = "relu2",
            [a + "ssm.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "ssm.state_size"] = g.Int("ssm.state_size"),
            [a + "ssm.group_count"] = g.Int("ssm.group_count"),
            [a + "ssm.head_count"] = heads,
            [a + "ssm.head_dim"] = g.Int("ssm.inner_size") / heads,
        };
        if (kv.Any(k => k > 0))
        {
            md[a + "attention.head_count"] = g.Int("attention.head_count");
            md[a + "attention.kv_head_count"] = kv.Max();
            md[a + "attention.head_dim"] = g.Int("attention.key_length", g.Int("embedding_length") / g.Int("attention.head_count"));
        }
        if (g.Has("context_length"))
        {
            md[a + "context_length"] = g.Int("context_length");
        }

        if (moeArch)
        {
            if (g.Int("expert_group_count", 1) > 1)
            {
                throw new NotSupportedException("nemotron_h_moe: group-limited expert routing is not supported yet.");
            }

            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.normalize_weights"] = g.Raw("expert_weights_norm") is not { } n || Convert.ToBoolean(n);
            if (g.Has("expert_weights_scale") && g.Float("expert_weights_scale") is var sc && sc != 1f)
            {
                md[a + "moe.weights_scale"] = sc;
            }
        }
        return md;
    }

    [GeneratedRegex(@"^blk\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

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

    public string? MapWeightName(string name, GgufMetadataView g)
    {
        switch (name)
        {
            case "token_embd.weight": return "token_embedding.weight";
            case "output_norm.weight": return "output_norm.weight";
            case "output.weight": return "lm_head.weight";
        }
        var m = BlockName().Match(name);
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
