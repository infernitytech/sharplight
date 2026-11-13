using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>mpt</c> (MPT-7B/30B): ALiBi (<c>attn_config.alibi_bias_max</c>) instead of a rotary embedding, layer norm (without bias when <c>no_bias</c>),
/// a fused [q; k; v] projection, optional q/k/v clamping (<c>clip_qkv</c>) and a plain exact-GELU MLP of <c>expansion_ratio</c> times the width.
/// Models with q/k layer norms (<c>qk_ln</c>) or without ALiBi are not supported.
/// </summary>
internal sealed partial class MptHfMapping : HfDecoderFamilyMapping
{
    private const int DefaultExpansionRatio = 4;
    private const float DefaultAlibiMaxBias = 8f;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["norm_1.weight"] = "input_norm.weight",
        ["norm_1.bias"] = "input_norm.bias",
        ["norm_2.weight"] = "post_attention_norm.weight",
        ["norm_2.bias"] = "post_attention_norm.bias",
        ["attn.out_proj.weight"] = "attention.o.weight",
        ["attn.out_proj.bias"] = "attention.o.bias",
        ["ffn.up_proj.weight"] = "mlp.up.weight",
        ["ffn.up_proj.bias"] = "mlp.up.bias",
        ["ffn.down_proj.weight"] = "mlp.down.weight",
        ["ffn.down_proj.bias"] = "mlp.down.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.wte.weight"] = "token_embedding.weight",
        ["transformer.norm_f.weight"] = "output_norm.weight",
        ["transformer.norm_f.bias"] = "output_norm.bias",
        ["wte.weight"] = "token_embedding.weight",
        ["norm_f.weight"] = "output_norm.weight",
        ["norm_f.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^(?:transformer\.)?blocks\.(\d+)\.(.+)$")]
    private static partial Regex MptLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["mpt"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => MptLayerName();

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "gelu";

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.Int("d_model", "hidden_size");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("n_heads", "num_attention_heads");

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("n_layers", "num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => (c.GetInt("expansion_ratio") ?? DefaultExpansionRatio) * HiddenSize(c);

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => AttentionConfigInt(c, "kv_n_heads") ?? headCount;

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("max_seq_len", "max_position_embeddings");

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        var attention = c.Get("attn_config") is { ValueKind: JsonValueKind.Object } element ? element : (JsonElement?)null;
        if (HfMappingHelpers.OptionalBool(attention, "qk_ln") == true)
        {
            throw new NotSupportedException("mpt: q/k layer norms (qk_ln) are not supported.");
        }

        if (HfMappingHelpers.OptionalBool(attention, "alibi") != true)
        {
            throw new NotSupportedException("mpt: only the ALiBi variant is supported.");
        }

        md["decoder.rope.layers"] = new int[LayerCount(c)];
        md["decoder.attention.alibi_max_bias"] = (float)HfMappingHelpers.NumberOr(attention, "alibi_bias_max", DefaultAlibiMaxBias);
        if (HfMappingHelpers.OptionalNumber(attention, "clip_qkv") is > 0 and var clip)
        {
            md["decoder.attention.clip_qkv"] = (float)clip;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix is not ("attn.Wqkv.weight" or "attn.Wqkv.bias"))
        {
            return false;
        }

        string kind = suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight";
        int heads = HeadCount(c);
        int headDim = HiddenSize(c) / heads;
        int kvRows = KvHeadCount(c, heads) * headDim;
        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, heads * headDim), $"{layerPrefix}attention.q.{kind}"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", heads * headDim, kvRows), $"{layerPrefix}attention.k.{kind}"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", heads * headDim + kvRows, kvRows), $"{layerPrefix}attention.v.{kind}"));
        return true;
    }

    private static int? AttentionConfigInt(HfConfig c, string key)
        => c.Get("attn_config") is { ValueKind: JsonValueKind.Object } attention && HfMappingHelpers.OptionalNumber(attention, key) is { } value ? (int)value : null;
}
