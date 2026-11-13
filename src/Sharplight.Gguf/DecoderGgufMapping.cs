using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp decoder-only architectures (<c>llama</c> (Llama, Mistral, SmolLM, ...), <c>qwen2</c>, <c>qwen3</c>,
/// <c>gemma2</c>, <c>gemma3</c>, <c>phi3</c> (fused qkv / gate+up are split), <c>olmoe</c> and <c>qwen3moe</c> (mixture of experts)) → canonical <c>decoder</c> (vocabulary documented on <c>DecoderAdapter</c>).
/// llama.cpp's "normal"-rope checkpoints (<c>llama</c>) store q/k rows permuted for interleaved rotation; they are
/// un-permuted lazily so canonical weights equal those of a Hugging Face import.
/// </summary>
public sealed partial class DecoderGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    public IReadOnlyList<string> GgufArchitectures { get; } = ["llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3", "olmoe", "qwen3moe"];
    public string Architecture => "decoder";

    static bool IsGemma(string a) => a is "gemma2" or "gemma3";
    static bool PermutesQk(string a) => a == "llama";

    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string arch = g.GgufArchitecture;
        string a = Architecture + ".";
        int nEmbd = g.Int("embedding_length");
        int nHead = g.Int("attention.head_count");
        int headDim = g.Int("attention.key_length", nEmbd / nHead);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count"),
            [a + "hidden_size"] = nEmbd,
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length", g.Int("expert_feed_forward_length", 0)),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv", nHead),
            [a + "attention.head_dim"] = headDim,
            [a + "rope.dim"] = g.Int("rope.dimension_count", headDim),
            [a + "rope.base"] = g.Has("rope.freq_base") ? g.Float("rope.freq_base") : 10000f,
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = IsGemma(arch) ? "gelu_tanh" : "silu",
        };
        if (g.Has("context_length")) md[a + "context_length"] = g.Int("context_length");
        if (arch is "olmoe" or "qwen3moe")
        {
            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.ffn_hidden_size"] = g.Int("expert_feed_forward_length", g.Int("feed_forward_length"));
            md[a + "moe.normalize_weights"] = arch == "qwen3moe";   // llama.cpp: expert_weights_norm
        }

        if (IsGemma(arch))
        {
            md[a + "embedding_scale"] = MathF.Sqrt(nEmbd);
            // llama.cpp: 1/sqrt(head_dim), except the 27B-class models (n_embd/n_head).
            bool big = arch == "gemma2" ? g.Int("block_count") == 46 : g.Int("block_count") == 62;
            md[a + "attention.scale"] = 1f / MathF.Sqrt(big ? nEmbd / (float)nHead : headDim);
            if (g.Has("attn_logit_softcapping")) md[a + "attention.logit_softcap"] = g.Float("attn_logit_softcapping");
            if (g.Has("final_logit_softcapping")) md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping");
            int window = g.Int("attention.sliding_window", 0);
            if (window > 0)
            {
                int pattern = arch == "gemma2" ? 2 : 6;   // llama.cpp: layer is sliding when il % pattern < pattern - 1
                int n = g.Int("block_count");
                md[a + "attention.sliding_window"] = window;
                md[a + "attention.sliding_layers"] = Enumerable.Range(0, n).Select(il => il % pattern < pattern - 1 ? 1 : 0).ToArray();
                if (arch == "gemma3") md[a + "rope.local_base"] = g.Has("rope.freq_base_swa") ? g.Float("rope.freq_base_swa") : 10000f;
            }
        }
        string? scaling = g.Raw("rope.scaling.type") as string;
        if (arch == "phi3")
        {
            // longrope: per-frequency factors come as tensors, the cos/sin magnitude as attn_factor.
            if (g.Has("rope.scaling.attn_factor") && g.Float("rope.scaling.attn_factor") is var af && af != 1f) md[a + "rope.attention_scale"] = af;
            if (g.Has("rope.scaling.original_context_length")) md[a + "rope.original_context_length"] = g.Int("rope.scaling.original_context_length");
        }
        else if (scaling == "yarn") md[a + "rope.attention_scale"] = YarnOf(g).AttentionScale;
        else if (scaling is not (null or "" or "none" or "linear")) throw new NotSupportedException($"GGUF rope.scaling.type '{scaling}' is not supported.");
        return md;
    }

    [GeneratedRegex(@"^blk\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

    static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q.bias"] = "attention.q.bias",
        ["attn_k.bias"] = "attention.k.bias",
        ["attn_v.bias"] = "attention.v.bias",
        ["attn_output.bias"] = "attention.o.bias",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        // Gemma 2/3: llama.cpp's post_attention_norm / post_ffw_norm are the norms of the sub-layer outputs.
        ["post_attention_norm.weight"] = "attention.out_norm.weight",
        ["post_ffw_norm.weight"] = "mlp.out_norm.weight",
    };

    public string? MapWeightName(string name, GgufMetadataView g)
    {
        switch (name)
        {
            case "token_embd.weight": return "token_embedding.weight";
            case "output_norm.weight": return "output_norm.weight";
            case "output.weight": return "lm_head.weight";
            case "rope_freqs.weight": return "rope.freq_factors";
            case "rope_factors_short.weight": return "rope.freq_factors";
            case "rope_factors_long.weight": return "rope.freq_factors_long";
        }
        var m = BlockName().Match(name);
        if (!m.Success) return null;
        return s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>Phi-3 stores q/k/v as one tensor and gate/up as one tensor (gate rows first).</summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        if (g.GgufArchitecture != "phi3") return null;
        var m = BlockName().Match(name);
        if (!m.Success) return null;
        string p = $"layers.{m.Groups[1].Value}.";
        int nHead = g.Int("attention.head_count"), nKv = g.Int("attention.head_count_kv", nHead);
        int headDim = g.Int("attention.key_length", g.Int("embedding_length") / nHead);
        switch (m.Groups[2].Value)
        {
            case "attn_qkv.weight":
                return [t.RowSlice(p + "attention.q.weight", 0, nHead * headDim),
                        t.RowSlice(p + "attention.k.weight", nHead * headDim, nKv * headDim),
                        t.RowSlice(p + "attention.v.weight", (nHead + nKv) * headDim, nKv * headDim)];
            case "ffn_up.weight":
                int ff = g.Int("feed_forward_length");
                if (t.Shape[0].Fixed != 2 * ff) return null;
                return [t.RowSlice(p + "mlp.gate.weight", 0, ff), t.RowSlice(p + "mlp.up.weight", ff, ff)];
            default: return null;
        }
    }

    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (!PermutesQk(g.GgufArchitecture)) return w;
        int nHead = g.Int("attention.head_count");
        if (w.Name.EndsWith("attention.q.weight") || w.Name.EndsWith("attention.q.bias")) return UnpermuteRows(w, nHead);
        if (w.Name.EndsWith("attention.k.weight") || w.Name.EndsWith("attention.k.bias")) return UnpermuteRows(w, g.Int("attention.head_count_kv", nHead));
        return w;
    }

    /// <summary>
    /// Inverts llama.cpp's convert-time permutation (per head: rows [a₀ b₀ a₁ b₁ …] back to [a₀ a₁ … b₀ b₁ …]).
    /// Rows are independent, so this is a byte-level row reorder for every weight type including quantized ones.
    /// </summary>
    static WeightInfo UnpermuteRows(WeightInfo w, int nHead)
    {
        long rows = w.Shape[0].Fixed;
        int headRows = checked((int)(rows / nHead)), half = headRows / 2;
        return WeightInfo.Derived(w.Name, w.Type, w.Shape, () =>
        {
            var src = w.ReadAll();
            int rowBytes = checked((int)(src.LongLength / rows));
            var dst = new byte[src.Length];
            for (int h = 0; h < nHead; h++)
                for (int r = 0; r < headRows; r++)
                {
                    int from = h * headRows + (r < half ? 2 * r : 2 * (r - half) + 1);
                    Buffer.BlockCopy(src, from * rowBytes, dst, (h * headRows + r) * rowBytes, rowBytes);
                }
            return dst;
        });
    }

    static RopeScaling.Result YarnOf(GgufMetadataView g)
    {
        int nHead = g.Int("attention.head_count");
        int dim = g.Int("rope.dimension_count", g.Int("attention.key_length", g.Int("embedding_length") / nHead));
        return RopeScaling.Yarn(dim, g.Has("rope.freq_base") ? g.Float("rope.freq_base") : 10000f, g.Float("rope.scaling.factor"),
            g.Int("rope.scaling.original_context_length"));
    }

    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        // Linear / YaRN scaling are described by metadata (llama3 scaling arrives as the stored rope_freqs.weight tensor).
        if (weights.ContainsKey("rope.freq_factors")) yield break;
        int nHead = g.Int("attention.head_count");
        int dim = g.Int("rope.dimension_count", g.Int("attention.key_length", g.Int("embedding_length") / nHead));
        RopeScaling.Result? r = null;
        if (g.Raw("rope.scaling.type") is "linear") r = RopeScaling.Linear(dim, g.Float("rope.scaling.factor"));
        else if (g.Raw("rope.scaling.type") is "yarn") r = YarnOf(g);
        if (r is null) yield break;
        var f = r.Factors;
        yield return WeightInfo.Derived("rope.freq_factors", DataType.F32, new Shape(f.Length), () =>
        {
            var b = new byte[f.Length * 4];
            Buffer.BlockCopy(f, 0, b, 0, b.Length);
            return b;
        });
    }
}
