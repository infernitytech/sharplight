using System.Text.RegularExpressions;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>exaone</c> (EXAONE 3.0/3.5, GPT-style tensor names): a Llama decoder whose gate is <c>mlp.c_fc_0</c> and up is <c>mlp.c_fc_1</c>.</summary>
internal sealed partial class ExaoneHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attn.attention.q_proj.weight"] = "attention.q.weight",
        ["attn.attention.k_proj.weight"] = "attention.k.weight",
        ["attn.attention.v_proj.weight"] = "attention.v.weight",
        ["attn.attention.out_proj.weight"] = "attention.o.weight",
        ["attn.attention.out_proj.bias"] = "attention.o.bias",
        ["mlp.c_fc_0.weight"] = "mlp.gate.weight",
        ["mlp.c_fc_1.weight"] = "mlp.up.weight",
        ["mlp.c_proj.weight"] = "mlp.down.weight",
        ["ln_1.weight"] = "input_norm.weight",
        ["ln_2.weight"] = "post_attention_norm.weight",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.wte.weight"] = "token_embedding.weight",
        ["transformer.ln_f.weight"] = "output_norm.weight",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^transformer\.h\.(\d+)\.(.+)$")]
    private static partial Regex ExaoneLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["exaone"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => ExaoneLayerName();

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("num_layers", "num_hidden_layers");
}
