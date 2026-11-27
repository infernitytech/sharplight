using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Salesforce Moirai 2.0 (the uni2ts package's <c>Moirai2Module</c>: Salesforce/moirai-2.0-R-small) → canonical <c>moirai2</c> (documented on
/// <c>MoiraiAdapter</c>). Like Moirai 1.x the config.json holds the module's keyword arguments without a <c>model_type</c> (recognized by
/// <see cref="MoiraiHfMapping.Detect"/>); the tensor names are kept.
/// </summary>
internal sealed class Moirai2HfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Moirai 2.0.</summary>
    public const string ModelType = "moirai2";

    /// <inheritdoc/>
    public string Architecture => ModelType;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("scaling") == false)
        {
            yield return "moirai2: checkpoints without scaling (scaling = false) are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        float[] levels = c.Root.TryGetProperty("quantile_levels", out var q) && q.ValueKind == JsonValueKind.Array
            ? [.. q.EnumerateArray().Select(x => (float)x.GetDouble())]
            : [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
        int hidden = c.Int("d_model");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "intermediate_size"] = c.GetInt("d_ff") ?? hidden * 4,
            [a + "layer_count"] = c.Int("num_layers"),
            [a + "patch_size"] = c.GetInt("patch_size") ?? 16,
            [a + "predict_tokens"] = c.GetInt("num_predict_token") ?? 1,
            [a + "quantiles"] = string.Join(',', levels.Select(x => x.ToString(CultureInfo.InvariantCulture))),
            [a + "max_sequence_length"] = c.GetInt("max_seq_len") ?? 512,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name));
        }

        return collector.Weights;
    }
}
