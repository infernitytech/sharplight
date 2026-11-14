using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>modernbert</c> (ModernBERT, modernbert-embed, Granite Embedding R2): a pre-norm bidirectional encoder with layer norms
/// (usually without biases), rotary embeddings, a GeGLU MLP and global attention every <c>global_attn_every_n_layers</c>-th layer (the first one
/// included) while the others attend within a symmetric local window. It maps onto the canonical decoder with <c>attention.causal</c> = false:
/// the window of <c>local_attention</c> keys becomes the exclusive bound <c>local_attention / 2 + 1</c>, local layers use
/// <c>local_rope_theta</c>, the fused <c>Wqkv</c> and <c>Wi</c> (activated half first) projections are split, and the first layer has no norm in
/// front of its attention. Masked-LM and classifier heads are not used.
/// </summary>
internal sealed partial class ModernBertHfMapping : HfDecoderFamilyMapping
{
    private const int DefaultGlobalEvery = 3;
    private const double DefaultGlobalRopeBase = 160000.0;
    private const double DefaultLocalRopeBase = 10000.0;
    private const int DefaultLocalAttention = 128;

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["attn_norm.bias"] = "input_norm.bias",
        ["attn.Wo.weight"] = "attention.o.weight",
        ["attn.Wo.bias"] = "attention.o.bias",
        ["mlp_norm.weight"] = "post_attention_norm.weight",
        ["mlp_norm.bias"] = "post_attention_norm.bias",
        ["mlp.Wo.weight"] = "mlp.down.weight",
        ["mlp.Wo.bias"] = "mlp.down.bias",
    };

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["model.embeddings.tok_embeddings.weight"] = "token_embedding.weight",
        ["model.embeddings.norm.weight"] = "embedding_norm.weight",
        ["model.embeddings.norm.bias"] = "embedding_norm.bias",
        ["model.final_norm.weight"] = "output_norm.weight",
        ["model.final_norm.bias"] = "output_norm.bias",
    };

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex ModernBertLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["modernbert"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layer;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_global;

    /// <inheritdoc/>
    protected override Regex LayerName => ModernBertLayerName();

    /// <inheritdoc/>
    protected override string BareTrunkMarker => "embeddings.tok_embeddings.weight";

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => (float)(c.GetDouble("norm_eps") ?? 1e-5);

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => (float)(c.LayerTypeRopeDouble("full_attention", "rope_theta") ?? c.GetDouble("global_rope_theta") ?? DefaultGlobalRopeBase);

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.StartsWith("head.", StringComparison.Ordinal) || name.StartsWith("decoder.", StringComparison.Ordinal)
           || name.StartsWith("classifier.", StringComparison.Ordinal) || name.StartsWith("model.head.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        int layers = LayerCount(c);
        int every = c.GetInt("global_attn_every_n_layers") ?? DefaultGlobalEvery;
        int[] sliding = c.GetStringArray("layer_types") is { } layerTypes
            ? HfMappingHelpers.SlidingFlags(layerTypes)
            : [.. Enumerable.Range(0, layers).Select(il => il % every == 0 ? 0 : 1)];
        md["decoder.attention.causal"] = false;
        md["decoder.attention.sliding_window"] = (c.GetInt("local_attention") ?? DefaultLocalAttention) / 2 + 1;
        md["decoder.attention.sliding_layers"] = sliding;
        md["decoder.rope.local_base"] = (float)(c.LayerTypeRopeDouble("sliding_attention", "rope_theta") ?? c.GetDouble("local_rope_theta") ?? DefaultLocalRopeBase);
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string parameter = suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight";
        int hidden = HiddenSize(c);
        int ffn = FeedForwardSize(c);
        switch (suffix)
        {
            case "attn.Wqkv.weight" or "attn.Wqkv.bias":
                collector.Add(HfTransforms.Plain(w.RowSlice($"{layerPrefix}attention.q.{parameter}", 0, hidden), $"{layerPrefix}attention.q.{parameter}"));
                collector.Add(HfTransforms.Plain(w.RowSlice($"{layerPrefix}attention.k.{parameter}", hidden, hidden), $"{layerPrefix}attention.k.{parameter}"));
                collector.Add(HfTransforms.Plain(w.RowSlice($"{layerPrefix}attention.v.{parameter}", 2 * hidden, hidden), $"{layerPrefix}attention.v.{parameter}"));
                return true;
            case "mlp.Wi.weight" or "mlp.Wi.bias":
                // chunk(2): the first half goes through the activation (the canonical gate), the second multiplies it.
                collector.Add(HfTransforms.Plain(w.RowSlice($"{layerPrefix}mlp.gate.{parameter}", 0, ffn), $"{layerPrefix}mlp.gate.{parameter}"));
                collector.Add(HfTransforms.Plain(w.RowSlice($"{layerPrefix}mlp.up.{parameter}", ffn, ffn), $"{layerPrefix}mlp.up.{parameter}"));
                return true;
            default:
                return false;
        }
    }
}
