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
internal sealed partial class EncoderHfMapping : IHfArchitectureMapping
{
    private const int DefaultPaddingIndex = 1;

    /// <summary>Prefixes the base model is saved under in task-specific checkpoints; dropped so names are uniform.</summary>
    private static readonly string[] s_modelPrefixes = ["bert.", "roberta.", "model."];

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
            AddNomicHyperParameters(md, a, c);
            return md;
        }

        md[a + "layer_count"] = c.Int("num_hidden_layers");
        md[a + "hidden_size"] = c.Int("hidden_size");
        md[a + "ffn_hidden_size"] = c.Int("intermediate_size");
        md[a + "attention.head_count"] = c.Int("num_attention_heads");
        md[a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-12);
        int maxPositions = c.GetInt("max_position_embeddings") ?? 512;
        md[a + "context_length"] = IsRoberta(c) ? maxPositions - (c.GetInt("pad_token_id") ?? DefaultPaddingIndex) - 1 : maxPositions;
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

    private static void AddNomicHyperParameters(Dictionary<string, object> md, string a, HfConfig c)
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

        string activation = c.GetString("activation_function") ?? "swiglu";
        if (activation is not ("swiglu" or "gelu"))
        {
            throw new NotSupportedException($"nomic_bert: activation '{activation}' is not supported.");
        }

        md[a + "mlp.activation"] = activation == "swiglu" ? "silu" : "gelu";
        md[a + "mlp.gated"] = activation == "swiglu";
        int headDim = c.Int("n_embd", "hidden_size") / c.Int("n_head", "num_attention_heads");
        md[a + "rope.dim"] = (int)(headDim * (c.GetDouble("rotary_emb_fraction") ?? 1.0));
        md[a + "rope.base"] = (float)(c.GetDouble("rotary_emb_base") ?? 10000.0);
        md[a + "rope.style"] = c.GetBool("rotary_emb_interleaved") == true ? "gptj" : "neox";
    }

    [GeneratedRegex(@"^(?:encoder\.layers?)\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>BERT / RoBERTa per-layer modules, HF module path to canonical module.</summary>
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

    /// <summary>nomic-bert per-layer modules, HF module path to canonical module.</summary>
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

    /// <summary>
    /// Embedding, pooler and classifier tensors outside the layers, HF name to canonical name. BertForSequenceClassification has a pooler
    /// (dense + tanh) then a classifier; RoBERTa has classifier.dense (+ tanh) then out_proj.
    /// </summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["embeddings.word_embeddings.weight"] = "token_embedding.weight",
        ["embeddings.token_type_embeddings.weight"] = "token_type_embedding.weight",
        ["embeddings.LayerNorm.weight"] = "embedding_norm.weight",
        ["emb_ln.weight"] = "embedding_norm.weight",
        ["embeddings.LayerNorm.bias"] = "embedding_norm.bias",
        ["emb_ln.bias"] = "embedding_norm.bias",
        ["pooler.dense.weight"] = "classifier.pre.weight",
        ["classifier.dense.weight"] = "classifier.pre.weight",
        ["pooler.dense.bias"] = "classifier.pre.bias",
        ["classifier.dense.bias"] = "classifier.pre.bias",
        ["classifier.weight"] = "classifier.out.weight",
        ["classifier.out_proj.weight"] = "classifier.out.weight",
        ["classifier.bias"] = "classifier.out.bias",
        ["classifier.out_proj.bias"] = "classifier.out.bias",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        int positionOffset = IsRoberta(c) ? (c.GetInt("pad_token_id") ?? DefaultPaddingIndex) + 1 : 0;
        int hidden = c.Int("n_embd", "hidden_size");
        bool isSequenceClassifier = c.Architectures.Any(x => x.Contains("SequenceClassification"));

        foreach (var (rawName, tensor) in tensors)
        {
            string name = NormalizeName(rawName);
            var info = tensor.ToWeightInfo();

            if (IsIgnored(name, isSequenceClassifier))
            {
                continue;   // buffers, masked-LM heads and the unused pooler of base models
            }

            if (s_global.TryGetValue(name, out var globalName))
            {
                collector.Add(info.As(globalName));
                continue;
            }

            if (name == "embeddings.position_embeddings.weight")
            {
                // RoBERTa counts positions from padding_idx + 1; canonical position 0 is the first real one.
                collector.Add(positionOffset == 0
                    ? info.As("position_embedding.weight")
                    : info.RowSlice("position_embedding.weight", positionOffset, info.Shape[0].Fixed - positionOffset));
                continue;
            }

            MapLayerTensor(c, rawName, name, info, hidden, collector);
        }

        return collector.Weights;
    }

    /// <summary>Drops the base-model prefix and renames old-style LayerNorm parameters (gamma / beta).</summary>
    private static string NormalizeName(string rawName)
    {
        string name = rawName;
        foreach (var prefix in s_modelPrefixes)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                name = name[prefix.Length..];
                break;
            }
        }

        if (name.EndsWith(".gamma", StringComparison.Ordinal))
        {
            return name[..^".gamma".Length] + ".weight";
        }

        if (name.EndsWith(".beta", StringComparison.Ordinal))
        {
            return name[..^".beta".Length] + ".bias";
        }

        return name;
    }

    private static bool IsIgnored(string name, bool isSequenceClassifier)
        => name.EndsWith("position_ids", StringComparison.Ordinal)
        || name.StartsWith("lm_head.", StringComparison.Ordinal)
        || name.StartsWith("cls.", StringComparison.Ordinal)
        || name.StartsWith("pooler.", StringComparison.Ordinal) && !isSequenceClassifier;

    private static void MapLayerTensor(HfConfig c, string rawName, string name, WeightInfo info, int hidden, HfWeightCollector collector)
    {
        var m = LayerName().Match(name);
        if (!m.Success)
        {
            throw new NotSupportedException($"{c.ModelType}: tensor '{rawName}' has no canonical mapping.");
        }

        string prefix = $"layers.{m.Groups[1].Value}.";
        string rest = m.Groups[2].Value;
        if (rest is "attn.Wqkv.weight" or "attn.Wqkv.bias")
        {
            // nomic-bert's fused qkv: q rows first, then k, then v.
            string parameter = rest["attn.Wqkv.".Length..];
            collector.Add(info.RowSlice(prefix + "attention.q." + parameter, 0, hidden));
            collector.Add(info.RowSlice(prefix + "attention.k." + parameter, hidden, hidden));
            collector.Add(info.RowSlice(prefix + "attention.v." + parameter, 2 * hidden, hidden));
            return;
        }

        int dot = rest.LastIndexOf('.');
        string module = rest[..dot];
        string parameterName = rest[(dot + 1)..];
        var table = IsNomic(c) ? s_nomicLayer : s_bertLayer;
        if (!table.TryGetValue(module, out var canonicalModule))
        {
            throw new NotSupportedException($"{c.ModelType}: tensor '{rawName}' has no canonical mapping.");
        }

        string canonicalName = prefix + canonicalModule + "." + parameterName;
        collector.Add(info.As(canonicalName));
    }
}
