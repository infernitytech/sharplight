using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>nemotron_h</c> checkpoints → canonical <c>nemotron_h</c> (vocabulary documented on <c>NemotronHAdapter</c>). A = -exp(A_log) and the conv are converted to f32 lazily.</summary>
public sealed partial class NemotronHHfMapping : IHfArchitectureMapping
{
    public string Architecture => "nemotron_h";
    public IReadOnlyList<string> ModelTypes { get; } = ["nemotron_h"];

    static string[] Kinds(HfConfig c, int nLayer)
    {
        if (c.GetString("hybrid_override_pattern") is { Length: > 0 } pat)
            return pat.Select(ch => ch switch
            {
                'M' => "ssm", '*' => "attention", '-' => "mlp", 'E' => "moe",
                _ => throw new NotSupportedException($"nemotron_h: unknown hybrid_override_pattern character '{ch}'."),
            }).ToArray();
        var list = c.GetStringArray("layers_block_type") ?? c.GetStringArray("layer_types") ?? throw new InvalidDataException("nemotron_h: no hybrid_override_pattern / layers_block_type in config.json.");
        return list.Select(t => t switch
        {
            "mamba" or "linear_attention" => "ssm", "attention" or "full_attention" => "attention", "mlp" => "mlp", "moe" => "moe",
            _ => throw new NotSupportedException($"nemotron_h: unknown layer type '{t}'."),
        }).ToArray();
    }

    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int nLayer = c.Int("num_hidden_layers");
        var kinds = Kinds(c, nLayer);
        if (kinds.Length != nLayer) throw new InvalidDataException("nemotron_h: layer pattern length differs from num_hidden_layers.");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon", "rms_norm_eps") ?? 1e-5),
            [a + "block_kinds"] = kinds,
            [a + "mlp.activation"] = (c.GetString("mlp_hidden_act") ?? "relu2") switch { "relu2" or "relu_squared" => "relu2", "silu" => "silu", "gelu" => "gelu", "relu" => "relu", var o => throw new NotSupportedException($"nemotron_h: activation '{o}'.") },
            [a + "ssm.conv_kernel"] = c.Int("conv_kernel"),
            [a + "ssm.state_size"] = c.Int("ssm_state_size"),
            [a + "ssm.group_count"] = c.Int("n_groups"),
            [a + "ssm.head_count"] = c.Int("mamba_num_heads"),
            [a + "ssm.head_dim"] = c.Int("mamba_head_dim"),
        };
        if (kinds.Contains("attention"))
        {
            int nHead = c.Int("num_attention_heads");
            md[a + "attention.head_count"] = nHead;
            md[a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? nHead;
            md[a + "attention.head_dim"] = c.GetInt("head_dim") ?? c.Int("hidden_size") / nHead;
        }
        if (c.GetInt("max_position_embeddings") is { } ctx) md[a + "context_length"] = ctx;
        if (kinds.Contains("moe")) throw new NotSupportedException("nemotron_h: mixture-of-experts layers are not supported yet.");
        return md;
    }

    [GeneratedRegex(@"^backbone\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["norm.weight"] = "input_norm.weight",
        ["mixer.in_proj.weight"] = "ssm.in.weight",
        ["mixer.out_proj.weight"] = "ssm.out.weight",
        ["mixer.q_proj.weight"] = "attention.q.weight",
        ["mixer.k_proj.weight"] = "attention.k.weight",
        ["mixer.v_proj.weight"] = "attention.v.weight",
        ["mixer.o_proj.weight"] = "attention.o.weight",
        ["mixer.up_proj.weight"] = "mlp.up.weight",
        ["mixer.down_proj.weight"] = "mlp.down.weight",
    };

    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var unmapped = new List<string>();
        void Add(WeightInfo w) { if (!result.TryAdd(w.Name, w)) throw new InvalidDataException($"Two HF tensors map to '{w.Name}'."); }
        foreach (var (name, t) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = t.ToWeightInfo();
            switch (name)
            {
                case "backbone.embeddings.weight" or "backbone.embedding.weight": Add(HfTransforms.Plain(w, "token_embedding.weight")); continue;
                case "backbone.norm_f.weight": Add(HfTransforms.Plain(w, "output_norm.weight")); continue;
                case "lm_head.weight":
                    if (!(c.GetBool("tie_word_embeddings") ?? false)) Add(HfTransforms.Plain(w, "lm_head.weight"));
                    continue;
            }
            var m = LayerName().Match(name);
            if (!m.Success) { unmapped.Add(name); continue; }
            string p = $"layers.{m.Groups[1].Value}.", suffix = m.Groups[2].Value;
            if (s_layer.TryGetValue(suffix, out var cn)) { Add(HfTransforms.Plain(w, p + cn)); continue; }
            switch (suffix)
            {
                case "mixer.A_log": Add(HfTransforms.ToF32(w, p + "ssm.a", map: static v => -MathF.Exp(v))); break;
                case "mixer.D": Add(HfTransforms.ToF32(w, p + "ssm.d")); break;
                case "mixer.dt_bias": Add(HfTransforms.ToF32(w, p + "ssm.dt_bias")); break;
                case "mixer.conv1d.weight": Add(HfTransforms.ToF32(w, p + "ssm.conv1d.weight", new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed))); break;
                case "mixer.conv1d.bias": Add(HfTransforms.ToF32(w, p + "ssm.conv1d.bias")); break;
                case "mixer.norm.weight": Add(HfTransforms.ToF32(w, p + "ssm.norm.weight")); break;
                default: unmapped.Add(name); break;
            }
        }
        if (unmapped.Count > 0)
            throw new NotSupportedException($"nemotron_h: {unmapped.Count} tensor(s) have no canonical mapping: {string.Join(", ", unmapped.Take(10))}{(unmapped.Count > 10 ? ", ..." : "")}");
        return result;
    }
}
