using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>bloom</c> (BLOOM, BLOOMZ): ALiBi position biases instead of a rotary embedding (maximum bias 8), layer norm with bias also on the embeddings,
/// fused per-head interleaved q/k/v with biases and a plain GELU MLP (tanh approximation). The models with the residual taken after the layer norm are not supported.
/// </summary>
internal sealed partial class BloomHfMapping : HfDecoderFamilyMapping
{
    private const float AlibiMaxBias = 8f;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
        ["self_attention.dense.weight"] = "attention.o.weight",
        ["self_attention.dense.bias"] = "attention.o.bias",
        ["mlp.dense_h_to_4h.weight"] = "mlp.up.weight",
        ["mlp.dense_h_to_4h.bias"] = "mlp.up.bias",
        ["mlp.dense_4h_to_h.weight"] = "mlp.down.weight",
        ["mlp.dense_4h_to_h.bias"] = "mlp.down.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.word_embeddings.weight"] = "token_embedding.weight",
        ["transformer.word_embeddings_layernorm.weight"] = "embedding_norm.weight",
        ["transformer.word_embeddings_layernorm.bias"] = "embedding_norm.bias",
        ["transformer.ln_f.weight"] = "output_norm.weight",
        ["transformer.ln_f.bias"] = "output_norm.bias",
        ["word_embeddings.weight"] = "token_embedding.weight",
        ["word_embeddings_layernorm.weight"] = "embedding_norm.weight",
        ["word_embeddings_layernorm.bias"] = "embedding_norm.bias",
        ["ln_f.weight"] = "output_norm.weight",
        ["ln_f.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^(?:transformer\.)?h\.(\d+)\.(.+)$")]
    private static partial Regex BloomLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["bloom"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => BloomLayerName();

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "gelu_tanh";

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.Int("hidden_size", "n_embed");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("n_head", "num_attention_heads");

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("n_layer", "num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => 4 * HiddenSize(c);

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => headCount;

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("seq_length", "max_position_embeddings");

    /// <inheritdoc/>
    protected override bool IsIgnored(string name) => base.IsIgnored(name) || name.EndsWith(".self_attention.bias", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("apply_residual_connection_post_layernorm") == true)
        {
            throw new NotSupportedException("bloom: apply_residual_connection_post_layernorm is not supported.");
        }

        md["decoder.rope.layers"] = new int[LayerCount(c)];
        md["decoder.attention.alibi_max_bias"] = AlibiMaxBias;
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix is not ("self_attention.query_key_value.weight" or "self_attention.query_key_value.bias"))
        {
            return false;
        }

        string kind = suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight";
        int heads = HeadCount(c);
        HfFusedProjections.SplitGroupedQkv(w, layerPrefix, kind, heads, heads, HiddenSize(c) / heads, collector);
        return true;
    }
}
