using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp bidirectional encoders (<c>bert</c> (BERT, MiniLM, bge, e5, and XLM-RoBERTa rerankers such as bge-reranker-v2-m3),
/// <c>nomic-bert</c> (rotary, SwiGLU, fused qkv)) → canonical <c>encoder</c> (vocabulary documented on <c>EncoderAdapter</c>).
/// llama.cpp's <c>pooling_type</c> becomes <see cref="CanonicalKeys.EmbeddingPooling"/>; the classifier head of ranking models
/// (<c>cls</c> dense + <c>cls.output</c>) maps to <c>classifier.pre</c> / <c>classifier.out</c>. The converter already removed
/// RoBERTa's position offset, so position row p is position p.
/// </summary>
internal sealed class EncoderGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["bert", "nomic-bert"];
    /// <inheritdoc/>
    public string Architecture => "encoder";

    private static bool IsNomic(GgufMetadataView g) => g.GgufArchitecture == "nomic-bert";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int embeddingLength = g.Int("embedding_length");
        int headCount = g.Int("attention.head_count");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count"),
            [a + "hidden_size"] = embeddingLength,
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "attention.head_count"] = headCount,
            [a + "norm_eps"] = g.Float("attention.layer_norm_epsilon"),
            [a + "mlp.activation"] = IsNomic(g) ? "silu" : "gelu",
        };
        if (IsNomic(g))
        {
            md[a + "mlp.gated"] = true;
            md[a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase);
            md[a + "rope.dim"] = g.Int("rope.dimension_count", embeddingLength / headCount);
            md[a + "rope.style"] = "neox";
        }

        GgufMappingHelpers.AddContextLength(md, a, g);

        if (g.Bool("attention.causal", false))
        {
            throw new NotSupportedException($"GGUF {g.GgufArchitecture}: causal encoders are not supported.");
        }
        // llama.cpp pooling_type: 0 none, 1 mean, 2 cls, 3 last, 4 rank (classifier on the cls row).
        string? pooling = g.Int("pooling_type", -1) switch { 0 => "none", 1 => "mean", 2 or 4 => "cls", 3 => "last", _ => null };
        if (pooling is not null)
        {
            md[CanonicalKeys.EmbeddingPooling] = pooling;
        }

        return md;
    }

    /// <summary>Per-layer tensors (without the <c>.weight</c> / <c>.bias</c> suffix), GGUF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_q"] = "attention.q",
        ["attn_k"] = "attention.k",
        ["attn_v"] = "attention.v",
        ["attn_output"] = "attention.o",
        ["attn_output_norm"] = "attention_norm",
        ["ffn_up"] = "mlp.up",
        ["ffn_gate"] = "mlp.gate",
        ["ffn_down"] = "mlp.down",
        ["layer_output_norm"] = "mlp_norm",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var globalName))
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string rest = m.Groups[2].Value;
        int dot = rest.LastIndexOf('.');
        if (dot < 0 || !s_layer.TryGetValue(rest[..dot], out var c))
        {
            return null;
        }

        return $"layers.{m.Groups[1].Value}.{c}{rest[dot..]}";
    }

    /// <summary>Embedding, position, classifier and embedding-norm tensors outside the layers.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["position_embd.weight"] = "position_embedding.weight",
        ["token_types.weight"] = "token_type_embedding.weight",
        ["token_embd_norm.weight"] = "embedding_norm.weight",
        ["token_embd_norm.bias"] = "embedding_norm.bias",
        ["cls.weight"] = "classifier.pre.weight",
        ["cls.bias"] = "classifier.pre.bias",
        ["cls.output.weight"] = "classifier.out.weight",
        ["cls.output.bias"] = "classifier.out.bias",
    };

    /// <summary>GGUF drops leading size-1 axes: a single token type or a one-label classifier comes as a vector; canonical form is [1, H].</summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
        => w.Shape.Rank == 1 && w.Name is "token_type_embedding.weight" or "classifier.out.weight" ? w.As(w.Name, new Shape(1, w.Shape[0].Fixed)) : w;

    /// <summary>nomic-bert stores q/k/v as one tensor (q rows first).</summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success || !m.Groups[2].Value.StartsWith("attn_qkv.", StringComparison.Ordinal))
        {
            return null;
        }

        string prefix = $"layers.{m.Groups[1].Value}.attention.";
        string parameter = m.Groups[2].Value["attn_qkv".Length..];
        int hidden = g.Int("embedding_length");
        return
        [
            t.RowSlice(prefix + "q" + parameter, 0, hidden),
            t.RowSlice(prefix + "k" + parameter, hidden, hidden),
            t.RowSlice(prefix + "v" + parameter, 2 * hidden, hidden),
        ];
    }
}
