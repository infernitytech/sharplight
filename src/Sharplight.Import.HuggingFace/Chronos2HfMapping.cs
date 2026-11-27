using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Chronos-2 (<c>Chronos2Model</c> of chronos-forecasting ≥ 2.0: a T5-style config with <c>model_type</c> <c>t5</c>, a <c>chronos_config</c> and
/// <c>chronos_pipeline_class</c> Chronos2Pipeline, e.g. amazon/chronos-2) → canonical <c>chronos2</c> (documented on <c>Chronos2Adapter</c>). Encoder block i
/// holds the time attention (layer.0), the group attention (layer.1) and the feed-forward block (layer.2). Only patches without overlap (stride = size =
/// output patch size) are supported.
/// </summary>
internal sealed partial class Chronos2HfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Chronos-2 checkpoints (their configs say <c>t5</c>; the <c>architectures</c> class decides).</summary>
    public const string ModelType = "chronos2";

    /// <inheritdoc/>
    public string Architecture => "chronos2";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (Chronos(c) is not { } chronos)
        {
            yield return "chronos2: config.json has no chronos_config.";
            yield break;
        }

        int size = Int(chronos, "input_patch_size", 16);
        if (Int(chronos, "input_patch_stride", size) != size || Int(chronos, "output_patch_size", size) != size)
        {
            yield return "chronos2: overlapping patches or an output patch size other than the input patch size are not supported.";
        }

        if (c.GetBool("is_gated_act") == true || (c.GetString("feed_forward_proj") ?? "relu").StartsWith("gated", StringComparison.Ordinal))
        {
            yield return "chronos2: gated feed-forward layers are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var chronos = Chronos(c) ?? throw new NotSupportedException("chronos2: config.json has no chronos_config.");
        string a = Architecture + ".";
        int hidden = c.GetInt("d_model") ?? 512;
        int context = Int(chronos, "context_length", 8192);
        float[] quantiles = chronos.TryGetProperty("quantiles", out var q) && q.ValueKind == JsonValueKind.Array
            ? [.. q.EnumerateArray().Select(x => (float)x.GetDouble())]
            : throw new NotSupportedException("chronos2: chronos_config has no quantiles.");
        string activation = (c.GetString("dense_act_fn") ?? (c.GetString("feed_forward_proj") ?? "relu").Split('-')[^1]) switch
        {
            "relu" => "relu",
            "gelu" => "gelu",
            "gelu_new" or "gelu_pytorch_tanh" => "gelu_tanh",
            "silu" or "swish" => "silu",
            var other => throw new NotSupportedException($"chronos2: dense_act_fn '{other}' is not supported."),
        };
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "intermediate_size"] = c.GetInt("d_ff") ?? 2048,
            [a + "layer_count"] = c.GetInt("num_layers") ?? 6,
            [a + "attention.head_count"] = c.GetInt("num_heads") ?? 8,
            [a + "attention.head_dim"] = c.GetInt("d_kv") ?? 64,
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon") ?? 1e-6),
            [a + "rope.base"] = (float)(c.GetDouble("rope_theta") ?? 10000.0),
            [a + "mlp.activation"] = activation,
            [a + "context_length"] = context,
            [a + "patch_size"] = Int(chronos, "input_patch_size", 16),
            [a + "max_output_patches"] = Int(chronos, "max_output_patches", 1),
            [a + "quantiles"] = string.Join(',', quantiles.Select(x => x.ToString(CultureInfo.InvariantCulture))),
            [a + "register_token"] = Bool(chronos, "use_reg_token", false),
            [a + "time_encoding_scale"] = Int(chronos, "time_encoding_scale", context),
            [a + "arcsinh"] = Bool(chronos, "use_arcsinh", false),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (name.StartsWith("input_patch_embedding.", StringComparison.Ordinal) || name.StartsWith("output_patch_embedding.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(info, name));
                continue;
            }

            switch (name)
            {
                case "shared.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "encoder.final_layer_norm.weight":
                    collector.Add(HfTransforms.Plain(info, "output_norm.weight"));
                    continue;
            }

            var m = BlockName().Match(name);
            string? module = !m.Success ? null : (m.Groups[2].Value, m.Groups[3].Value) switch
            {
                ("0", "layer_norm.weight") => "time_attention_norm.weight",
                ("0", var rest) when Projection().IsMatch(rest) => "time_attention." + rest["self_attention.".Length..],
                ("1", "layer_norm.weight") => "group_attention_norm.weight",
                ("1", var rest) when Projection().IsMatch(rest) => "group_attention." + rest["self_attention.".Length..],
                ("2", "layer_norm.weight") => "mlp_norm.weight",
                ("2", "mlp.wi.weight") => "mlp.up.weight",
                ("2", "mlp.wo.weight") => "mlp.down.weight",
                _ => null,
            };
            if (module is null)
            {
                throw new NotSupportedException($"chronos2: tensor '{name}' has no canonical mapping.");
            }

            string canonical = $"layers.{m.Groups[1].Value}.{module}";
            collector.Add(module.EndsWith("norm.weight", StringComparison.Ordinal) ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        return collector.Weights;
    }

    [GeneratedRegex(@"^encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

    [GeneratedRegex(@"^self_attention\.[qkvo]\.weight$")]
    private static partial Regex Projection();

    private static JsonElement? Chronos(HfConfig c) => c.Get("chronos_config") is { ValueKind: JsonValueKind.Object } v ? v : null;

    private static int Int(JsonElement e, string key, int fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static bool Bool(JsonElement e, string key, bool fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.ValueKind == JsonValueKind.True : fallback;
}
