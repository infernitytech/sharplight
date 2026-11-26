using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// TimesFM 1 / 2 in the transformers layout (<c>TimesFmModelForPrediction</c>, model_type <c>timesfm</c>, e.g. google/timesfm-2.0-500m-pytorch) → canonical
/// <c>timesfm</c> (documented on <c>TimesFmAdapter</c>): the <c>decoder.</c> prefix is dropped, the other names are kept. Checkpoints with the sinusoidal position
/// embedding (<c>use_positional_embedding</c>) are not supported. TimesFM 2.5 checkpoints of the timesfm package also say <c>timesfm</c>; they go to
/// <see cref="TimesFm25HfMapping"/>.
/// </summary>
internal sealed class TimesFmHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "timesfm";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["timesfm"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("use_positional_embedding") == true)
        {
            yield return "timesfm: the sinusoidal position embedding (use_positional_embedding) is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.GetInt("hidden_size") ?? 1280;
        int heads = c.GetInt("num_attention_heads") ?? 16;
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "intermediate_size"] = c.GetInt("intermediate_size") ?? hidden,
            [a + "layer_count"] = c.GetInt("num_hidden_layers") ?? 50,
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = c.GetInt("head_dim") ?? (hidden / heads),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "context_length"] = c.GetInt("context_length") ?? 2048,
            [a + "prediction_length"] = c.GetInt("horizon_length") ?? 128,
            [a + "patch_size"] = c.GetInt("patch_length") ?? 32,
            [a + "quantiles"] = Quantiles(c),
            [a + "decode_index"] = 0,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name.StartsWith("decoder.", StringComparison.Ordinal) ? name["decoder.".Length..] : name));
        }

        return collector.Weights;
    }

    /// <summary>The <c>quantiles</c> of a TimesFM config, comma separated (default 0.1 ... 0.9).</summary>
    internal static string Quantiles(HfConfig c)
    {
        float[] levels = c.Get("quantiles") is { ValueKind: JsonValueKind.Array } q
            ? [.. q.EnumerateArray().Select(x => (float)x.GetDouble())]
            : [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
        return string.Join(',', levels.Select(x => x.ToString(CultureInfo.InvariantCulture)));
    }
}
