using System.Text.Json;
using System.Text.RegularExpressions;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>embedding_gemma2</c> / <c>embedding_gemma2_text</c> (EmbeddingGemma 2): a bidirectional Gemma 4-style encoder. Sliding layers
/// attend within the inclusive radius <c>sliding_window</c> (the canonical exclusive bound is one more), full-attention layers have their own head
/// size and key/value head count (<c>per_layer_config</c>, per-layer head sizes), q/k/v are RMS-normed (v without weight), attention is unscaled, every
/// layer adds a per-layer embedding that comes from a projection of the token embeddings alone (no per-layer token table) and ends with
/// <c>layer_scalar</c>. The final <c>embedding_projection</c> becomes the pooled projection <c>embedding_projection.0</c> (projecting before mean
/// pooling is the same). Vision and audio towers are not imported.
/// </summary>
internal sealed partial class EmbeddingGemma2HfMapping : HfDecoderFamilyMapping
{
    private const double DefaultGlobalRopeBase = 1000000.0;
    private const double DefaultLocalRopeBase = 10000.0;

    private static readonly IReadOnlyDictionary<string, string> s_layer = With(LlamaLayerTensors,
        ("post_attention_layernorm.weight", "attention.out_norm.weight"),
        ("pre_feedforward_layernorm.weight", "post_attention_norm.weight"),
        ("post_feedforward_layernorm.weight", "mlp.out_norm.weight"),
        ("ple_block.per_layer_input_gate.weight", "per_layer.gate.weight"),
        ("ple_block.per_layer_projection.weight", "per_layer.projection.weight"),
        ("ple_block.post_per_layer_input_norm.weight", "per_layer.norm.weight"),
        ("layer_scalar", "output_scale"));

    private static readonly IReadOnlyDictionary<string, string> s_global = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.embed_tokens.weight"] = "token_embedding.weight",
        ["model.norm.weight"] = "output_norm.weight",
        ["model.ple.per_layer_model_projection.weight"] = "per_layer.model_projection.weight",
        ["model.ple.per_layer_projection_norm.weight"] = "per_layer.projection_norm.weight",
        ["model.embedding_projection.weight"] = "embedding_projection.0.weight",
        ["model.language_model.embed_tokens.weight"] = "token_embedding.weight",
        ["model.language_model.norm.weight"] = "output_norm.weight",
        ["model.language_model.ple.per_layer_model_projection.weight"] = "per_layer.model_projection.weight",
        ["model.language_model.ple.per_layer_projection_norm.weight"] = "per_layer.projection_norm.weight",
        ["model.language_model.embedding_projection.weight"] = "embedding_projection.0.weight",
    };

    // The text model's tensors, saved alone (EmbeddingGemma2TextModel, bare or under "model.") or inside the composite model ("model.language_model.").
    [GeneratedRegex(@"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerNameRegex();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["embedding_gemma2", "embedding_gemma2_text"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layer;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_global;

    /// <inheritdoc/>
    protected override Regex LayerName => LayerNameRegex();

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => (float)(c.GetDouble("rms_norm_eps") ?? 1e-6);

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => (float)(c.LayerTypeRopeDouble("full_attention", "rope_theta") ?? DefaultGlobalRopeBase);

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.StartsWith("model.vision_tower.", StringComparison.Ordinal) || name.StartsWith("model.audio_tower.", StringComparison.Ordinal)
           || name.StartsWith("model.embed_vision.", StringComparison.Ordinal) || name.StartsWith("model.embed_audio.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string a = "decoder.";
        int layers = LayerCount(c);
        int hidden = HiddenSize(c);
        int headDim = HeadDimOf(c);
        int kvHeads = KvHeadCount(c, HeadCount(c));
        string[] types = c.GetStringArray("layer_types") ?? [.. Enumerable.Repeat("full_attention", layers)];
        var headDims = new int[layers];
        var kvHeadCounts = new int[layers];
        for (int il = 0; il < layers; il++)
        {
            var layer = LayerConfig(c, il);
            headDims[il] = layer is { } l && l.TryGetProperty("head_dim", out var dim) ? dim.GetInt32() : headDim;
            kvHeadCounts[il] = layer is { } k && k.TryGetProperty("num_key_value_heads", out var kv) ? kv.GetInt32() : kvHeads;
        }

        int globalDim = Enumerable.Range(0, layers).Where(il => types[il] == "full_attention").Select(il => headDims[il]).DefaultIfEmpty(headDim).First();
        md[a + "attention.causal"] = false;
        md[a + "attention.scale"] = 1f;
        md[a + "attention.v_norm"] = true;
        md[a + "attention.layer_head_dims"] = headDims;
        md[a + "attention.layer_kv_head_counts"] = kvHeadCounts;
        md[a + "attention.sliding_window"] = (c.GetInt("sliding_window") ?? 512) + 1;
        md[a + "attention.sliding_layers"] = HfMappingHelpers.SlidingFlags(types);
        md[a + "rope.dim"] = globalDim;
        md[a + "rope.local_dim"] = headDim;
        md[a + "rope.local_base"] = (float)(c.LayerTypeRopeDouble("sliding_attention", "rope_theta") ?? DefaultLocalRopeBase);
        md[a + "embedding_scale"] = MathF.Sqrt(hidden);
        md[a + "per_layer.size"] = c.Int("hidden_size_per_layer_input");
    }

    // The per_layer_config entry of a layer ({"head_dim", "num_key_value_heads"}), keyed by the layer index as a string.
    private static JsonElement? LayerConfig(HfConfig c, int layer)
        => c.Get("per_layer_config") is { ValueKind: JsonValueKind.Object } perLayer && perLayer.TryGetProperty(layer.ToString(System.Globalization.CultureInfo.InvariantCulture), out var entry)
            ? entry : null;
}
