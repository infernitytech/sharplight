using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Chronos (the original token forecasters of the chronos-forecasting package: <c>T5ForConditionalGeneration</c> checkpoints whose <c>config.json</c> has a
/// <c>chronos_config</c> with the <c>MeanScaleUniformBins</c> tokenizer, e.g. amazon/chronos-t5-small) → canonical <c>chronos</c> (documented on
/// <c>ChronosAdapter</c>): the T5 hyper-parameters and weights of <see cref="T5HfMapping"/> under the <c>chronos.</c> prefix, the decoder's cross-attention cached
/// per source, plus the tokenizer and sampling settings.
/// </summary>
internal sealed class ChronosHfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Chronos checkpoints (their configs say <c>t5</c>; the <c>chronos_config</c> decides).</summary>
    public const string ModelType = "chronos";

    private static readonly T5HfMapping s_t5 = new();

    /// <inheritdoc/>
    public string Architecture => "chronos";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <summary>Whether <paramref name="config"/> is a Chronos token forecaster (a T5 config with a <c>chronos_config</c> that names a tokenizer class).</summary>
    public static bool IsTokenModel(HfConfig config)
        => config.Root.ValueKind == JsonValueKind.Object && config.Root.TryGetProperty("chronos_config", out var chronos) && chronos.ValueKind == JsonValueKind.Object
           && chronos.TryGetProperty("tokenizer_class", out _);

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        var chronos = Chronos(c);
        if (String(chronos, "tokenizer_class") != "MeanScaleUniformBins")
        {
            yield return $"chronos: tokenizer class '{String(chronos, "tokenizer_class")}' is not supported (only MeanScaleUniformBins).";
        }

        if ((String(chronos, "model_type") ?? "seq2seq") != "seq2seq")
        {
            yield return "chronos: only the encoder-decoder (seq2seq) models are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var chronos = Chronos(c);
        string a = Architecture + ".";
        var result = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in s_t5.MapHyperParameters(c, options))
        {
            result[a + key[(s_t5.Architecture.Length + 1)..]] = value;
        }

        int context = Int(chronos, "context_length", 512);
        bool eos = !chronos.TryGetProperty("use_eos_token", out var useEos) || useEos.ValueKind != JsonValueKind.False;
        var tokenizer = chronos.TryGetProperty("tokenizer_kwargs", out var kwargs) && kwargs.ValueKind == JsonValueKind.Object ? kwargs : default;
        result[a + "cross_attention.cached"] = true;
        result[a + "encoder.rows"] = context + (eos ? 1 : 0);
        result[a + "encoder.variable_rows"] = true;
        result[a + "encoder.key_bias"] = true;   // missing values inside the context are masked out of the encoder's attention (#93)
        result[a + "forecast.context_length"] = context;
        result[a + "forecast.prediction_length"] = Int(chronos, "prediction_length", 64);
        result[a + "forecast.tokens"] = Int(chronos, "n_tokens", 4096);
        result[a + "forecast.special_tokens"] = Int(chronos, "n_special_tokens", 2);
        result[a + "forecast.pad_token"] = Int(chronos, "pad_token_id", 0);
        result[a + "forecast.end_token"] = eos ? Int(chronos, "eos_token_id", 1) : -1;
        result[a + "forecast.low_limit"] = (float)Double(tokenizer, "low_limit", -15);
        result[a + "forecast.high_limit"] = (float)Double(tokenizer, "high_limit", 15);
        result[a + "forecast.sample_count"] = Int(chronos, "num_samples", 20);
        result[a + "forecast.temperature"] = (float)Double(chronos, "temperature", 1);
        result[a + "forecast.top_k"] = Int(chronos, "top_k", 50);
        return result;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
        => s_t5.MapWeights(c, tensors, options);

    private static JsonElement Chronos(HfConfig c) => c.Get("chronos_config") is { ValueKind: JsonValueKind.Object } v ? v : default;

    private static string? String(JsonElement e, string key)
        => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int Int(JsonElement e, string key, int fallback)
        => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static double Double(JsonElement e, string key, double fallback)
        => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;
}
