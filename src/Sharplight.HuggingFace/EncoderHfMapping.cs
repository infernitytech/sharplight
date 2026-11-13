using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face bidirectional encoders (<c>bert</c>, <c>roberta</c>, <c>xlm-roberta</c>, <c>nomic_bert</c>) → canonical <c>encoder</c>
/// (vocabulary documented on <c>EncoderAdapter</c>). Base models (<c>BertModel</c>, embedding checkpoints) and sequence
/// classifiers (<c>*ForSequenceClassification</c>, cross-encoder rerankers) are both handled; model prefixes (<c>bert.</c>,
/// <c>roberta.</c>) are dropped, RoBERTa's position table is cut to start at position 0 (its first padding_idx + 1 rows are never
/// used), nomic-bert's fused qkv is split zero-copy.
/// </summary>
public sealed partial class EncoderHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "encoder";
    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["bert", "roberta", "xlm-roberta", "nomic_bert"];

    private static bool IsRoberta(HfConfig c) => c.ModelType is "roberta" or "xlm-roberta";
    private static bool IsNomic(HfConfig c) => c.ModelType == "nomic_bert";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        if (IsNomic(c))
        {
            md[a + "layer_count"] = c.Int("n_layer", "num_hidden_layers");
            md[a + "hidden_size"] = c.Int("n_embd", "hidden_size");
            md[a + "ffn_hidden_size"] = c.Int("n_inner", "intermediate_size");
            md[a + "attention.head_count"] = c.Int("n_head", "num_attention_heads");
            md[a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon", "layer_norm_eps") ?? 1e-12);
            md[a + "context_length"] = c.GetInt("max_trained_positions", "n_positions") ?? 2048;
            if (c.GetBool("prenorm") == true)
            {
                throw new NotSupportedException("nomic_bert: pre-norm checkpoints are not supported.");
            }

            string act = c.GetString("activation_function") ?? "swiglu";
            if (act is not ("swiglu" or "gelu"))
            {
                throw new NotSupportedException($"nomic_bert: activation '{act}' is not supported.");
            }

            md[a + "mlp.activation"] = act == "swiglu" ? "silu" : "gelu";
            md[a + "mlp.gated"] = act == "swiglu";
            int headDim = c.Int("n_embd", "hidden_size") / c.Int("n_head", "num_attention_heads");
            md[a + "rope.dim"] = (int)(headDim * (c.GetDouble("rotary_emb_fraction") ?? 1.0));
            md[a + "rope.base"] = (float)(c.GetDouble("rotary_emb_base") ?? 10000.0);
            md[a + "rope.style"] = c.GetBool("rotary_emb_interleaved") == true ? "gptj" : "neox";
            return md;
        }
        md[a + "layer_count"] = c.Int("num_hidden_layers");
        md[a + "hidden_size"] = c.Int("hidden_size");
        md[a + "ffn_hidden_size"] = c.Int("intermediate_size");
        md[a + "attention.head_count"] = c.Int("num_attention_heads");
        md[a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-12);
        int maxPos = c.GetInt("max_position_embeddings") ?? 512;
        md[a + "context_length"] = IsRoberta(c) ? maxPos - (c.GetInt("pad_token_id") ?? 1) - 1 : maxPos;
        md[a + "mlp.activation"] = (c.GetString("hidden_act") ?? "gelu") switch
        {
            "gelu" => "gelu",
            "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
            "relu" => "relu",
            "silu" or "swish" => "silu",
            var other => throw new NotSupportedException($"{c.ModelType}: activation '{other}' is not supported."),
        };
        if ((c.GetString("position_embedding_type") ?? "absolute") != "absolute")
        {
            throw new NotSupportedException($"{c.ModelType}: position_embedding_type '{c.GetString("position_embedding_type")}' is not supported.");
        }

        return md;
    }

    [GeneratedRegex(@"^(?:encoder\.layers?)\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    private static readonly Dictionary<string, string> s_bertLayer = new(StringComparer.Ordinal)
    {
        ["attention.self.query"] = "attention.q",
        ["attention.self.key"] = "attention.k",
        ["attention.self.value"] = "attention.v",
        ["attention.output.dense"] = "attention.o",
        ["attention.output.LayerNorm"] = "attention_norm",
        ["intermediate.dense"] = "mlp.up",
        ["output.dense"] = "mlp.down",
        ["output.LayerNorm"] = "mlp_norm",
    };

    private static readonly Dictionary<string, string> s_nomicLayer = new(StringComparer.Ordinal)
    {
        ["attn.out_proj"] = "attention.o",
        ["norm1"] = "attention_norm",
        ["mlp.fc11"] = "mlp.up",
        ["mlp.fc12"] = "mlp.gate",
        ["mlp.fc1"] = "mlp.up",
        ["mlp.fc2"] = "mlp.down",
        ["norm2"] = "mlp_norm",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var w = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        int posOffset = IsRoberta(c) ? (c.GetInt("pad_token_id") ?? 1) + 1 : 0;
        int nEmbd = c.Int("n_embd", "hidden_size");
        void Add(string name, WeightInfo info) { if (!w.TryAdd(name, info)) { throw new InvalidDataException($"Two tensors map to '{name}'."); } }

        foreach (var (raw, t) in tensors)
        {
            string name = raw;
            foreach (var prefix in new[] { "bert.", "roberta.", "model." })
            {
                if (name.StartsWith(prefix, StringComparison.Ordinal)) { name = name[prefix.Length..]; break; }
            }
            // Old checkpoints call LayerNorm parameters gamma / beta.
            if (name.EndsWith(".gamma", StringComparison.Ordinal))
            {
                name = name[..^6] + ".weight";
            }
            else if (name.EndsWith(".beta", StringComparison.Ordinal))
            {
                name = name[..^5] + ".bias";
            }

            var info = t.ToWeightInfo();

            if (name.EndsWith("position_ids", StringComparison.Ordinal) || name.StartsWith("lm_head.", StringComparison.Ordinal)
                || name.StartsWith("cls.", StringComparison.Ordinal) || name.StartsWith("pooler.", StringComparison.Ordinal) && !c.Architectures.Any(x => x.Contains("SequenceClassification")))
            {
                continue;   // buffers, masked-LM heads and the unused pooler of base models
            }

            switch (name)
            {
                case "embeddings.word_embeddings.weight": Add("token_embedding.weight", info.As("token_embedding.weight")); continue;
                case "embeddings.token_type_embeddings.weight": Add("token_type_embedding.weight", info.As("token_type_embedding.weight")); continue;
                case "embeddings.position_embeddings.weight":
                    Add("position_embedding.weight", posOffset == 0 ? info.As("position_embedding.weight")
                        : info.RowSlice("position_embedding.weight", posOffset, info.Shape[0].Fixed - posOffset));
                    continue;
                case "embeddings.LayerNorm.weight" or "emb_ln.weight": Add("embedding_norm.weight", info.As("embedding_norm.weight")); continue;
                case "embeddings.LayerNorm.bias" or "emb_ln.bias": Add("embedding_norm.bias", info.As("embedding_norm.bias")); continue;
                // BertForSequenceClassification: pooler (dense + tanh) then classifier; RoBERTa: classifier.dense (+ tanh) then out_proj.
                case "pooler.dense.weight" or "classifier.dense.weight": Add("classifier.pre.weight", info.As("classifier.pre.weight")); continue;
                case "pooler.dense.bias" or "classifier.dense.bias": Add("classifier.pre.bias", info.As("classifier.pre.bias")); continue;
                case "classifier.weight" or "classifier.out_proj.weight": Add("classifier.out.weight", info.As("classifier.out.weight")); continue;
                case "classifier.bias" or "classifier.out_proj.bias": Add("classifier.out.bias", info.As("classifier.out.bias")); continue;
            }
            var m = LayerName().Match(name);
            if (!m.Success)
            {
                throw new NotSupportedException($"{c.ModelType}: tensor '{raw}' has no canonical mapping.");
            }

            string p = $"layers.{m.Groups[1].Value}.", rest = m.Groups[2].Value;
            if (rest == "attn.Wqkv.weight")
            {
                Add(p + "attention.q.weight", info.RowSlice(p + "attention.q.weight", 0, nEmbd));
                Add(p + "attention.k.weight", info.RowSlice(p + "attention.k.weight", nEmbd, nEmbd));
                Add(p + "attention.v.weight", info.RowSlice(p + "attention.v.weight", 2 * nEmbd, nEmbd));
                continue;
            }
            if (rest == "attn.Wqkv.bias")
            {
                Add(p + "attention.q.bias", info.RowSlice(p + "attention.q.bias", 0, nEmbd));
                Add(p + "attention.k.bias", info.RowSlice(p + "attention.k.bias", nEmbd, nEmbd));
                Add(p + "attention.v.bias", info.RowSlice(p + "attention.v.bias", 2 * nEmbd, nEmbd));
                continue;
            }
            int dot = rest.LastIndexOf('.');
            string module = rest[..dot], param = rest[(dot + 1)..];
            var table = IsNomic(c) ? s_nomicLayer : s_bertLayer;
            if (!table.TryGetValue(module, out var canon))
            {
                throw new NotSupportedException($"{c.ModelType}: tensor '{raw}' has no canonical mapping.");
            }

            Add(p + canon + "." + param, info.As(p + canon + "." + param));
        }
        return w;
    }
}
