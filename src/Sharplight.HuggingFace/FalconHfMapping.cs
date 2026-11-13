using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>falcon</c> (Falcon 7B/40B/180B, RW): a fused q/k/v tensor, layer norm with bias, plain GELU MLP and, usually, parallel residual. Three fused layouts
/// exist: the new decoder architecture groups per key/value head, multi-query stores all queries then one key and one value, the legacy layout interleaves per head.
/// The new architecture has two norms (<c>ln_attn</c>, <c>ln_mlp</c>). ALiBi is not supported.
/// </summary>
internal sealed partial class FalconHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["self_attention.dense.weight"] = "attention.o.weight",
        ["self_attention.dense.bias"] = "attention.o.bias",
        ["mlp.dense_h_to_4h.weight"] = "mlp.up.weight",
        ["mlp.dense_h_to_4h.bias"] = "mlp.up.bias",
        ["mlp.dense_4h_to_h.weight"] = "mlp.down.weight",
        ["mlp.dense_4h_to_h.bias"] = "mlp.down.bias",
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["ln_attn.weight"] = "input_norm.weight",
        ["ln_attn.bias"] = "input_norm.bias",
        ["ln_mlp.weight"] = "post_attention_norm.weight",
        ["ln_mlp.bias"] = "post_attention_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.word_embeddings.weight"] = "token_embedding.weight",
        ["transformer.ln_f.weight"] = "output_norm.weight",
        ["transformer.ln_f.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^transformer\.h\.(\d+)\.(.+)$")]
    private static partial Regex FalconLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["falcon", "RefinedWebModel", "RefinedWeb"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => FalconLayerName();

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => c.GetString("activation") is null or "gelu" ? "gelu" : base.Activation(c);

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("num_hidden_layers", "n_layer");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("num_attention_heads", "n_head");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("ffn_hidden_size") ?? 4 * HiddenSize(c);

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount)
    {
        if (c.GetBool("new_decoder_architecture") == true)
        {
            return c.GetInt("num_kv_heads", "n_head_kv") ?? headCount;
        }

        return c.GetBool("multi_query") == true ? 1 : headCount;
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("alibi") == true)
        {
            throw new NotSupportedException("falcon: ALiBi position biases are not supported.");
        }

        bool newArchitecture = c.GetBool("new_decoder_architecture") == true;
        md["decoder.residual.parallel"] = newArchitecture || (c.GetBool("parallel_attn") ?? true);
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
        int kvHeads = KvHeadCount(c, heads);
        int headDim = HeadDimOf(c);
        if (c.GetBool("new_decoder_architecture") == true || c.GetBool("multi_query") != true)
        {
            // Grouped per key/value head; with one query per group this is the legacy per-head interleaving.
            HfFusedProjections.SplitGroupedQkv(w, layerPrefix, kind, heads, kvHeads, headDim, collector);
            return true;
        }

        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, heads * headDim), $"{layerPrefix}attention.q.{kind}"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", heads * headDim, headDim), $"{layerPrefix}attention.k.{kind}"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", (heads + 1) * headDim, headDim), $"{layerPrefix}attention.v.{kind}"));
        return true;
    }
}
