using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>opt</c> (Meta's OPT, <c>OPTForCausalLM</c>): learned absolute position embeddings stored with an offset of two
/// rows (row <c>p + 2</c> is position <c>p</c>; the first two rows are dropped), pre-norm layer norm with bias, separate q/k/v
/// projections with bias and a non-gated MLP (<c>fc1</c> / <c>fc2</c>, ReLU by default). The post-norm variant
/// (<c>do_layer_norm_before = false</c>, OPT-350m) and the projected embeddings (<c>word_embed_proj_dim</c> other than the hidden
/// size) are refused by <see cref="Validate"/>.
/// </summary>
internal sealed partial class OptHfMapping : HfDecoderFamilyMapping
{
    /// <summary>Rows in front of position 0 in <c>embed_positions</c> (transformers' <c>OPTLearnedPositionalEmbedding.offset</c>).</summary>
    private const int PositionOffset = 2;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.out_proj.weight"] = "attention.o.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.out_proj.bias"] = "attention.o.bias",
        ["self_attn_layer_norm.weight"] = "input_norm.weight",
        ["self_attn_layer_norm.bias"] = "input_norm.bias",
        ["final_layer_norm.weight"] = "post_attention_norm.weight",
        ["final_layer_norm.bias"] = "post_attention_norm.bias",
        ["fc1.weight"] = "mlp.up.weight",
        ["fc1.bias"] = "mlp.up.bias",
        ["fc2.weight"] = "mlp.down.weight",
        ["fc2.bias"] = "mlp.down.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.decoder.embed_tokens.weight"] = "token_embedding.weight",
        ["model.decoder.final_layer_norm.weight"] = "output_norm.weight",
        ["model.decoder.final_layer_norm.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^model\.decoder\.layers\.(\d+)\.(.+)$")]
    private static partial Regex OptLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["opt"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => OptLayerName();

    /// <inheritdoc/>
    protected override string BareTrunkMarker => "decoder.embed_tokens.weight";

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.Int("ffn_dim");

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        if (config.GetBool("do_layer_norm_before") == false)
        {
            yield return "OPT with post-layer norm (do_layer_norm_before = false, e.g. OPT-350m) is not supported.";
        }

        if (config.GetInt("word_embed_proj_dim") is { } projection && projection != config.Int("hidden_size"))
        {
            yield return "OPT with projected embeddings (word_embed_proj_dim other than hidden_size) is not supported.";
        }

        if (config.GetBool("enable_bias") == false)
        {
            yield return "OPT without biases (enable_bias = false) is not supported.";
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.rope.layers"] = new int[LayerCount(c)];
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        if (name != "model.decoder.embed_positions.weight")
        {
            return false;
        }

        long rows = w.Shape[0].Fixed - PositionOffset;
        collector.Add(HfTransforms.Plain(w.RowSlice("", PositionOffset, rows), "position_embedding.weight"));
        return true;
    }
}
