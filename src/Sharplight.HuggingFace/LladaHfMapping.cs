using System.Text.RegularExpressions;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>llada</c> (LLaDA 8B, remote code <c>LLaDAModelLM</c>): Llama layers with bidirectional attention under OLMo-style names
/// (<c>model.transformer.blocks.{i}.*</c>, <c>d_model</c> / <c>n_heads</c> / <c>mlp_hidden_size</c> config keys), a masked-diffusion language
/// model whose logits row i predicts position i (canonical <c>attention.causal</c> = false, <c>diffusion.shift_logits</c> = false).
/// </summary>
internal sealed partial class LladaHfMapping : HfDecoderFamilyMapping
{
    /// <summary>Per-block tensors, LLaDA suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["q_proj.weight"] = "attention.q.weight",
        ["k_proj.weight"] = "attention.k.weight",
        ["v_proj.weight"] = "attention.v.weight",
        ["attn_out.weight"] = "attention.o.weight",
        ["ff_proj.weight"] = "mlp.gate.weight",
        ["up_proj.weight"] = "mlp.up.weight",
        ["ff_out.weight"] = "mlp.down.weight",
        ["attn_norm.weight"] = "input_norm.weight",
        ["ff_norm.weight"] = "post_attention_norm.weight",
    };

    /// <summary>Tensors outside the blocks.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["model.transformer.wte.weight"] = "token_embedding.weight",
        ["model.transformer.ln_f.weight"] = "output_norm.weight",
        ["model.transformer.ff_out.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^model\.transformer\.blocks\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["llada"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layer;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_global;

    /// <inheritdoc/>
    protected override Regex LayerName => BlockName();

    /// <inheritdoc/>
    protected override string BareTrunkMarker => "transformer.wte.weight";

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.GetInt("d_model") ?? c.Int("hidden_size");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.GetInt("n_heads") ?? c.Int("num_attention_heads");

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => c.GetInt("n_kv_heads", "num_key_value_heads") ?? headCount;

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.GetInt("n_layers") ?? c.Int("num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("mlp_hidden_size") ?? c.Int("intermediate_size");

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("max_sequence_length", "max_position_embeddings");

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c) => DreamHfMapping.AddDiffusion(md, c, shiftLogits: false);
}
