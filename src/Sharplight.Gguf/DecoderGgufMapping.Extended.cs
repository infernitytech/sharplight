using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>gemma4</c> (and <c>muse-glimmer</c>): the generic decoder plus per-layer head sizes, KV sharing, per-layer embeddings, NoPE layers, ...</summary>
public sealed partial class DecoderGgufMapping
{
    static bool IsExtended(string a) => a is "gemma4" or "muse-glimmer";

    static readonly Dictionary<string, string> s_extended = new(StringComparer.Ordinal)
    {
        ["inp_gate.weight"] = "per_layer.gate.weight",
        ["proj.weight"] = "per_layer.projection.weight",
        ["post_norm.weight"] = "per_layer.norm.weight",
        ["layer_output_scale.weight"] = "output_scale",
        ["attn_gate.weight"] = "attention.gate.weight",
    };

    static readonly Dictionary<string, string> s_extendedGlobal = new(StringComparer.Ordinal)
    {
        ["per_layer_token_embd.weight"] = "per_layer.embedding.weight",
        ["per_layer_model_proj.weight"] = "per_layer.model_projection.weight",
        ["per_layer_proj_norm.weight"] = "per_layer.projection_norm.weight",
    };

    /// <summary>A per-layer GGUF array (or a scalar repeated) as ints.</summary>
    static int[] PerLayer(GgufMetadataView g, string key, int nLayer, int fallback)
    {
        if (g.Raw(key) is Array arr && arr.Length == nLayer) return Enumerable.Range(0, nLayer).Select(i => Convert.ToInt32(arr.GetValue(i))).ToArray();
        return Enumerable.Repeat(g.Has(key) && g.Raw(key) is not Array ? g.Int(key) : fallback, nLayer).ToArray();
    }

    IReadOnlyDictionary<string, object> Gemma4Hyper(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int nLayer = g.Int("block_count"), nHead = g.Int("attention.head_count"), nEmbd = g.Int("embedding_length");
        int globalDim = g.Int("attention.key_length"), localDim = g.Int("attention.key_length_swa", globalDim);
        // llama.cpp: sliding_window_pattern[il] = true for sliding layers.
        var slid = g.Raw("attention.sliding_window_pattern") as Array ?? throw new InvalidDataException("gemma4: attention.sliding_window_pattern missing.");
        var sliding = Enumerable.Range(0, nLayer).Select(i => Convert.ToBoolean(slid.GetValue(i)) ? 1 : 0).ToArray();
        int kv = PerLayer(g, "attention.head_count_kv", nLayer, nHead)[0];
        var kvPer = PerLayer(g, "attention.head_count_kv", nLayer, nHead);
        var ffn = PerLayer(g, "feed_forward_length", nLayer, 0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = nEmbd,
            [a + "ffn_hidden_size"] = ffn.Min(),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = kv,
            [a + "attention.head_dim"] = localDim,
            [a + "attention.scale"] = 1f,
            [a + "attention.v_norm"] = true,
            [a + "attention.layer_head_dims"] = sliding.Select(s => s != 0 ? localDim : globalDim).ToArray(),
            [a + "attention.layer_kv_head_counts"] = kvPer,
            [a + "attention.sliding_window"] = g.Int("attention.sliding_window", 0),
            [a + "attention.sliding_layers"] = sliding,
            [a + "rope.dim"] = g.Int("rope.dimension_count", globalDim),
            [a + "rope.local_dim"] = g.Int("rope.dimension_count_swa", localDim),
            [a + "rope.base"] = g.Has("rope.freq_base") ? g.Float("rope.freq_base") : 1000000f,
            [a + "rope.local_base"] = g.Has("rope.freq_base_swa") ? g.Float("rope.freq_base_swa") : 10000f,
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = "gelu_tanh",
            [a + "embedding_scale"] = MathF.Sqrt(nEmbd),
        };
        if (g.Has("context_length")) md[a + "context_length"] = g.Int("context_length");
        if (g.Has("final_logit_softcapping")) md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping");
        int shared = g.Int("attention.shared_kv_layers", 0);
        if (shared > 0)
        {
            int first = nLayer - shared;
            var source = new int[nLayer];
            for (int il = 0; il < nLayer; il++)
                source[il] = il < first ? il : Enumerable.Range(0, first).Last(j => sliding[j] == sliding[il]);
            md[a + "attention.kv_source_layers"] = source;
        }
        if (g.Int("embedding_length_per_layer_input", 0) is var ple and > 0) md[a + "per_layer.size"] = ple;
        return md;
    }
}
