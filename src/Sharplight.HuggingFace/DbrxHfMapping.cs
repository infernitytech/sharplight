using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>dbrx</c> (Databricks DBRX): layer norms without bias, a fused [q; k; v] projection clamped to ±<c>clip_qkv</c>, rotary
/// embedding, and a softmax router over SwiGLU experts whose top-k weights are renormalized (<c>moe_normalize_expert_weights</c> = 1, the L1 norm).
/// The experts are stored fused as [E·F, H] matrices (<c>w1</c> gate, <c>v1</c> up, <c>w2</c> down, the last one transposed).
/// </summary>
internal sealed partial class DbrxHfMapping : HfMoeFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["norm_attn_norm.norm_1.weight"] = "input_norm.weight",
        ["norm_attn_norm.norm_2.weight"] = "post_attention_norm.weight",
        ["norm_attn_norm.attn.out_proj.weight"] = "attention.o.weight",
        ["ffn.router.layer.weight"] = "mlp.router.weight",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.wte.weight"] = "token_embedding.weight",
        ["transformer.norm_f.weight"] = "output_norm.weight",
        ["lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^transformer\.blocks\.(\d+)\.(.+)$")]
    private static partial Regex DbrxLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["dbrx"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => DbrxLayerName();

    /// <inheritdoc/>
    protected override string Activation(HfConfig c)
        => SubConfig(c, "ffn_config") is { } ffn && ffn.TryGetProperty("ffn_act_fn", out var act) && act.ValueKind == JsonValueKind.Object
           && act.TryGetProperty("name", out var name) && name.GetString() is { } activation && activation != "silu"
            ? throw new NotSupportedException($"dbrx: expert activation '{activation}' is not supported.")
            : "silu";

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.Int("d_model", "hidden_size");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("n_heads", "num_attention_heads");

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("n_layers", "num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => FfnInt(c, "ffn_hidden_size");

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => (int)(HfMappingHelpers.OptionalNumber(SubConfig(c, "attn_config"), "kv_n_heads") ?? 1);

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("max_seq_len", "max_position_embeddings");

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => 1e-5f;

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c)
        => (float)(c.RopeDouble("rope_theta", "rope_theta") ?? HfMappingHelpers.OptionalNumber(SubConfig(c, "attn_config"), "rope_theta") ?? 10000.0);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
    {
        double? p = HfMappingHelpers.OptionalNumber(SubConfig(c, "ffn_config"), "moe_normalize_expert_weights");
        if (SubConfig(c, "ffn_config") is { } ffn && ffn.TryGetProperty("moe_normalize_expert_weights", out var value) && value.ValueKind == JsonValueKind.Null)
        {
            p = null;
        }
        else
        {
            p ??= 1.0;
        }

        if (p is { } norm && norm != 1.0)
        {
            throw new NotSupportedException($"dbrx: moe_normalize_expert_weights = {norm} (only the L1 norm) is not supported.");
        }

        return new(FfnInt(c, "moe_num_experts"), FfnInt(c, "moe_top_k"), FfnInt(c, "ffn_hidden_size"), "softmax", Normalize: p is not null);
    }

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (HfMappingHelpers.OptionalNumber(SubConfig(c, "attn_config"), "clip_qkv") is > 0 and var clip)
        {
            md["decoder.attention.clip_qkv"] = (float)clip;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        int hidden = HiddenSize(c);
        switch (suffix)
        {
            case "norm_attn_norm.attn.Wqkv.weight":
                int heads = HeadCount(c);
                int headDim = hidden / heads;
                int kvRows = KvHeadCount(c, heads) * headDim;
                collector.Add(HfTransforms.Plain(w.RowSlice("", 0, heads * headDim), layerPrefix + "attention.q.weight"));
                collector.Add(HfTransforms.Plain(w.RowSlice("", heads * headDim, kvRows), layerPrefix + "attention.k.weight"));
                collector.Add(HfTransforms.Plain(w.RowSlice("", heads * headDim + kvRows, kvRows), layerPrefix + "attention.v.weight"));
                return true;
            case "ffn.experts.mlp.w1":
            case "ffn.experts.mlp.v1":
            case "ffn.experts.mlp.w2":
                var experts = w.As(w.Name, new Shape(FfnInt(c, "moe_num_experts"), FfnInt(c, "ffn_hidden_size"), hidden));
                collector.Add(suffix[^2..] switch
                {
                    "w1" => HfTransforms.Plain(experts, layerPrefix + "mlp.experts.gate.weight"),
                    "v1" => HfTransforms.Plain(experts, layerPrefix + "mlp.experts.up.weight"),
                    _ => HfExpertLayouts.Transposed(experts, layerPrefix + "mlp.experts.down.weight"),
                });
                return true;
            default:
                return false;
        }
    }

    private static JsonElement? SubConfig(HfConfig c, string key) => c.Get(key) is { ValueKind: JsonValueKind.Object } element ? element : null;

    private static int FfnInt(HfConfig c, string key)
        => (int)(HfMappingHelpers.OptionalNumber(SubConfig(c, "ffn_config"), key) ?? throw new InvalidDataException($"dbrx: ffn_config.{key} is missing."));
}
