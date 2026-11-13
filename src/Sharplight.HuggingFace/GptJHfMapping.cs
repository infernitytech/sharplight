using System.Text.RegularExpressions;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>gptj</c> (GPT-J): interleaved partial rotary, layer norm with bias, attention and MLP in parallel around one norm, plain biased GELU MLP, biased output head.</summary>
internal sealed partial class GptJHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attn.q_proj.weight"] = "attention.q.weight",
        ["attn.k_proj.weight"] = "attention.k.weight",
        ["attn.v_proj.weight"] = "attention.v.weight",
        ["attn.out_proj.weight"] = "attention.o.weight",
        ["mlp.fc_in.weight"] = "mlp.up.weight",
        ["mlp.fc_in.bias"] = "mlp.up.bias",
        ["mlp.fc_out.weight"] = "mlp.down.weight",
        ["mlp.fc_out.bias"] = "mlp.down.bias",
        ["ln_1.weight"] = "input_norm.weight",
        ["ln_1.bias"] = "input_norm.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.wte.weight"] = "token_embedding.weight",
        ["transformer.ln_f.weight"] = "output_norm.weight",
        ["transformer.ln_f.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
        ["lm_head.bias"] = "lm_head.bias",
    };

    [GeneratedRegex(@"^transformer\.h\.(\d+)\.(.+)$")]
    private static partial Regex GptJLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["gptj"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => GptJLayerName();

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.Int("n_embd", "hidden_size");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("n_head", "num_attention_heads");

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("n_layer", "num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("n_inner") ?? 4 * HiddenSize(c);

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => headCount;

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("n_positions", "max_position_embeddings");

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim) => c.GetInt("rotary_dim") ?? headDim;

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.EndsWith(".attn.bias", StringComparison.Ordinal) || name.EndsWith(".attn.masked_bias", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.residual.parallel"] = true;
    }
}
