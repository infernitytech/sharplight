using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Chronos-Bolt (<c>ChronosBoltModelForForecasting</c> of the chronos-forecasting package: a T5 config with <c>model_type</c> <c>t5</c> and a <c>chronos_config</c>,
/// e.g. amazon/chronos-bolt-small) → canonical <c>chronos_bolt</c> (documented on <c>ChronosBoltAdapter</c>). The T5 encoder and decoder map as for
/// <see cref="T5HfMapping"/>; the input and output residual blocks keep their names. Only patches without overlap (stride = size) are supported.
/// </summary>
internal sealed class ChronosBoltHfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Chronos-Bolt checkpoints (their configs say <c>t5</c>; the <c>architectures</c> class decides).</summary>
    public const string ModelType = "chronos_bolt";

    private static readonly T5HfMapping s_t5 = new();

    /// <inheritdoc/>
    public string Architecture => "chronos_bolt";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (Chronos(c) is not { } chronos)
        {
            yield return "chronos_bolt: config.json has no chronos_config.";
            yield break;
        }

        int size = Int(chronos, "input_patch_size", 16);
        if (Int(chronos, "input_patch_stride", size) != size)
        {
            yield return "chronos_bolt: overlapping patches (input_patch_stride != input_patch_size) are not supported.";
        }

        if (c.GetBool("is_gated_act") == true)
        {
            yield return "chronos_bolt: gated feed-forward layers are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var chronos = Chronos(c) ?? throw new NotSupportedException("chronos_bolt: config.json has no chronos_config.");
        string a = Architecture + ".";
        var result = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in s_t5.MapHyperParameters(c, options))
        {
            result[a + key[(s_t5.Architecture.Length + 1)..]] = value;
        }

        float[] quantiles = chronos.TryGetProperty("quantiles", out var q) && q.ValueKind == JsonValueKind.Array
            ? [.. q.EnumerateArray().Select(x => (float)x.GetDouble())]
            : [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
        result[a + "context_length"] = Int(chronos, "context_length", 2048);
        result[a + "prediction_length"] = Int(chronos, "prediction_length", 64);
        result[a + "patch_size"] = Int(chronos, "input_patch_size", 16);
        result[a + "quantiles"] = string.Join(',', quantiles.Select(x => x.ToString(CultureInfo.InvariantCulture)));
        result[a + "register_token"] = !chronos.TryGetProperty("use_reg_token", out var reg) || reg.ValueKind != JsonValueKind.False;
        result[a + "patch.activation"] = (c.GetString("dense_act_fn") ?? "relu") switch
        {
            "relu" => "relu",
            "gelu" => "gelu",
            "gelu_new" or "gelu_pytorch_tanh" => "gelu_tanh",
            "silu" or "swish" => "silu",
            var other => throw new NotSupportedException($"chronos_bolt: dense_act_fn '{other}' is not supported."),
        };
        return result;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var t5 = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith("input_patch_embedding.", StringComparison.Ordinal) || name.StartsWith("output_patch_embedding.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name));
            }
            else if (name != "lm_head.weight")
            {
                t5[name] = tensor;
            }
        }

        foreach (var (_, weight) in s_t5.MapWeights(c, t5, options))
        {
            collector.Add(weight);
        }

        return collector.Weights;
    }

    private static JsonElement? Chronos(HfConfig c) => c.Get("chronos_config") is { ValueKind: JsonValueKind.Object } v ? v : null;

    private static int Int(JsonElement e, string key, int fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
}
