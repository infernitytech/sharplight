using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// Everything architecture-specific about importing one GGUF (llama.cpp) architecture: the canonical
/// architecture id, the hyper-parameter map and the tensor name map. llama.cpp's naming stays inside this
/// assembly; adding a model family is one class implementing this interface plus a
/// <see cref="GgufArchitectureMappings.Register"/> call.
/// </summary>
public interface IGgufArchitectureMapping
{
    /// <summary>GGUF <c>general.architecture</c> values handled (e.g. "qwen35").</summary>
    public IReadOnlyList<string> GgufArchitectures { get; }

    /// <summary>Canonical architecture id (e.g. "qwen3_5").</summary>
    public string Architecture { get; }

    /// <summary>Canonical <c>{arch}.*</c> hyper-parameters (full keys) from the GGUF metadata.</summary>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView gguf);

    /// <summary>Canonical name of a GGUF tensor, or null when it has none (the import then fails).</summary>
    public string? MapWeightName(string ggufName, GgufMetadataView gguf);

    /// <summary>
    /// Optional value/layout transform of a tensor that already carries its canonical name (e.g. a lazily derived row
    /// permutation), so canonical weights are format independent. Default: unchanged (zero-copy).
    /// </summary>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf) => weight;

    /// <summary>Canonical weights that GGUF does not store but the metadata determines (e.g. computed rope frequency factors).</summary>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights) => [];

    /// <summary>
    /// The canonical architecture id for this particular file. Mappings whose GGUF architecture covers several
    /// layouts (e.g. <c>clip</c> projector types) override it; default: <see cref="Architecture"/>.
    /// </summary>
    public string ResolveArchitecture(GgufMetadataView gguf) => Architecture;
}

/// <summary>
/// Optional second stage of a mapping: rewrites the canonically named weight set where the file layout needs
/// several tensors combined into one canonical weight (derived lazily).
/// </summary>
public interface IGgufWeightTransformer
{
    public void Transform(Dictionary<string, Models.WeightInfo> weights, GgufMetadataView gguf);
}

/// <summary>Optional capability of a mapping: one GGUF tensor that becomes several canonical weights (fused projections).</summary>
public interface IGgufSplitMapping
{
    /// <summary>The canonical weights a GGUF tensor splits into, or null when the tensor maps one-to-one.</summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf);
}

/// <summary>Read access to GGUF metadata of one architecture (<c>{gguf_arch}.key</c>).</summary>
public sealed class GgufMetadataView(IReadOnlyDictionary<string, object> metadata, string ggufArchitecture)
{
    public IReadOnlyDictionary<string, object> All => metadata;
    public string GgufArchitecture => ggufArchitecture;
    public bool Has(string key) => metadata.ContainsKey($"{ggufArchitecture}.{key}");
    public object? Raw(string key) => metadata.TryGetValue($"{ggufArchitecture}.{key}", out var v) ? v : null;
    public int Int(string key) => Convert.ToInt32(Raw(key) ?? throw new InvalidDataException($"GGUF metadata {ggufArchitecture}.{key} missing."));
    public int Int(string key, int fallback) => Raw(key) is { } v ? Convert.ToInt32(v) : fallback;
    public float Float(string key) => Convert.ToSingle(Raw(key) ?? throw new InvalidDataException($"GGUF metadata {ggufArchitecture}.{key} missing."));
}

/// <summary>Registry of <see cref="IGgufArchitectureMapping"/>s keyed by GGUF architecture id.</summary>
public static class GgufArchitectureMappings
{
    private static readonly ConcurrentDictionary<string, IGgufArchitectureMapping> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    static GgufArchitectureMappings()
    {
        Register(new Qwen35GgufMapping());
        Register(new NemotronHGgufMapping());
        Register(new ClipGgufMapping());
        Register(new DecoderGgufMapping());
        Register(new EncoderGgufMapping());
    }

    public static void Register(IGgufArchitectureMapping mapping)
    {
        foreach (var a in mapping.GgufArchitectures)
        {
            s_byArch[a] = mapping;
        }
    }

    public static IEnumerable<IGgufArchitectureMapping> All => s_byArch.Values.Distinct();

    public static IGgufArchitectureMapping? Find(string? ggufArchitecture)
        => ggufArchitecture != null && s_byArch.TryGetValue(ggufArchitecture, out var m) ? m : null;
}

/// <summary>llama.cpp <c>qwen35</c> (Qwen3.5 / 3.6 / 3.8 dense hybrid) → canonical <c>qwen3_5</c>.</summary>
public sealed partial class Qwen35GgufMapping : IGgufArchitectureMapping
{
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen35"];
    public string Architecture => "qwen3_5";

    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        int nextn = g.Int("nextn_predict_layers", 0);
        int valueHeads = g.Int("ssm.time_step_rank");
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count") - nextn,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = g.Int("attention.head_count"),
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv"),
            [a + "attention.head_dim"] = g.Int("attention.key_length"),
            [a + "rope.dim"] = g.Int("rope.dimension_count"),
            [a + "full_attention_interval"] = g.Int("full_attention_interval", 4),
            [a + "linear_attention.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "linear_attention.key_head_dim"] = g.Int("ssm.state_size"),
            [a + "linear_attention.key_head_count"] = g.Int("ssm.group_count"),
            [a + "linear_attention.value_head_count"] = valueHeads,
            [a + "linear_attention.value_head_dim"] = g.Int("ssm.inner_size") / valueHeads,
            [a + "mtp.layer_count"] = nextn,
        };
        if (g.Has("context_length"))
        {
            md[a + "context_length"] = g.Int("context_length");
        }

        if (g.Has("rope.freq_base"))
        {
            md[a + "rope.base"] = g.Float("rope.freq_base");
        }

        if (g.Raw("rope.dimension_sections") is Array sec)
        {
            var s = new int[4];
            for (int i = 0; i < Math.Min(4, sec.Length); i++)
            {
                s[i] = Convert.ToInt32(sec.GetValue(i));
            }

            md[a + "rope.sections"] = s;
        }
        return md;
    }

    [GeneratedRegex(@"^blk\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_qkv.weight"] = "linear_attention.qkv.weight",
        ["attn_gate.weight"] = "linear_attention.z.weight",
        ["ssm_beta.weight"] = "linear_attention.beta.weight",
        ["ssm_alpha.weight"] = "linear_attention.alpha.weight",
        ["ssm_out.weight"] = "linear_attention.out.weight",
        ["ssm_norm.weight"] = "linear_attention.norm.weight",
        ["ssm_conv1d.weight"] = "linear_attention.conv1d.weight",
        ["ssm_a"] = "linear_attention.a",
        ["ssm_dt.bias"] = "linear_attention.dt_bias",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
    };

    private static readonly Dictionary<string, string> s_nextn = new(StringComparer.Ordinal)
    {
        ["nextn.eh_proj.weight"] = "fc.weight",
        ["nextn.enorm.weight"] = "embedding_norm.weight",
        ["nextn.hnorm.weight"] = "hidden_norm.weight",
        ["nextn.shared_head_norm.weight"] = "output_norm.weight",
        ["nextn.shared_head_head.weight"] = "lm_head.weight",
        ["nextn.embed_tokens.weight"] = "token_embedding.weight",
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
        if (!m.Success)
        {
            return null;
        }

        int il = int.Parse(m.Groups[1].Value);
        string suffix = m.Groups[2].Value;
        int nLayer = g.Int("block_count") - g.Int("nextn_predict_layers", 0);
        if (il < nLayer)
        {
            return s_layer.TryGetValue(suffix, out var c) ? $"layers.{il}.{c}" : null;
        }
        // MTP (NextN) layers: per-layer tensors under mtp.layers.{j}, the shared ones under mtp.*.
        if (s_nextn.TryGetValue(suffix, out var n))
        {
            return $"mtp.{n}";
        }

        return s_layer.TryGetValue(suffix, out var l) ? $"mtp.layers.{il - nLayer}.{l}" : null;
    }
}
