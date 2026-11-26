namespace Sharplight.TimeSeries;

/// <summary>
/// The graph contracts of the <see cref="Models.ModelKinds.TimeSeriesForecasting"/> kind. Two forms exist:
/// <list type="bullet">
/// <item><description>Patch-quantile models (Chronos-Bolt): one <see cref="Models.ModelComponentRoles.Encoder"/> component with the graph <see cref="ForecastGraph"/>.
/// The host scales the context (instance normalization: mean and standard deviation of the observed values), left-pads it with missing values to
/// <see cref="ContextLengthKey"/> observations and cuts it into patches of <see cref="PatchSizeKey"/> values; per patch the input row holds the scaled values (0 where
/// missing) followed by the observation mask (1 observed, 0 missing). The graph returns the scaled forecast of every quantile level of
/// <see cref="QuantilesKey"/>, <see cref="PredictionLengthKey"/> steps each.</description></item>
/// <item><description>Token models (Chronos): an <see cref="Models.ModelComponentRoles.Encoder"/> and a <see cref="Models.ModelComponentRoles.Decoder"/> component
/// (the encoder-decoder <c>encode</c> / <c>decode</c> graphs of a text model) over a vocabulary of value bins: the host mean-scales the context, maps every value to the
/// bin whose interval holds it (<see cref="BinCountKey"/> bins with centers spread evenly over [<see cref="LowLimitKey"/>, <see cref="HighLimitKey"/>] after
/// <see cref="SpecialTokensKey"/> special tokens), samples the forecast token by token and maps the tokens back to the bin centers.</description></item>
/// </list>
/// The configuration keys sit on the encoder component.
/// </summary>
public static class TimeSeriesGraphs
{
    /// <summary>The patch-quantile graph: <see cref="PatchesInput"/> and <see cref="MaskInput"/> in, <see cref="QuantilesOutput"/> out.</summary>
    public const string ForecastGraph = "forecast";

    /// <summary>The patches, f32 <c>[patches, 2 · patch size]</c>: per patch the scaled values (0 where missing), then the observation mask.</summary>
    public const string PatchesInput = "patches";

    /// <summary>The patch mask, f32 <c>[patches]</c>: 1 where the patch has at least one observed value, 0 where it has none (those patches are not attended).</summary>
    public const string MaskInput = "patch_mask";

    /// <summary>The scaled forecast, f32 <c>[quantile levels, prediction length]</c>.</summary>
    public const string QuantilesOutput = "quantiles";

    /// <summary>The configuration key of the form: <see cref="PatchQuantileForm"/> or <see cref="TokenForm"/>.</summary>
    public const string FormKey = "form";

    /// <summary>The <see cref="FormKey"/> of patch-quantile models.</summary>
    public const string PatchQuantileForm = "patch-quantile";

    /// <summary>The <see cref="FormKey"/> of token models.</summary>
    public const string TokenForm = "tokens";

    /// <summary>The configuration key of the observations the model looks at.</summary>
    public const string ContextLengthKey = "contextLength";

    /// <summary>The configuration key of the steps one pass predicts.</summary>
    public const string PredictionLengthKey = "predictionLength";

    /// <summary>The configuration key of the quantile levels of a patch-quantile model (comma separated, invariant culture).</summary>
    public const string QuantilesKey = "quantiles";

    /// <summary>The configuration key of the values per patch.</summary>
    public const string PatchSizeKey = "patchSize";

    /// <summary>The configuration key of the vocabulary size of a token model (special tokens included).</summary>
    public const string BinCountKey = "tokens";

    /// <summary>The configuration key of the special tokens before the first bin (pad, end of sequence).</summary>
    public const string SpecialTokensKey = "specialTokens";

    /// <summary>The configuration key of the lowest bin center (in units of the mean scale).</summary>
    public const string LowLimitKey = "lowLimit";

    /// <summary>The configuration key of the highest bin center.</summary>
    public const string HighLimitKey = "highLimit";

    /// <summary>The configuration key of the padding token (missing values).</summary>
    public const string PadTokenKey = "padToken";

    /// <summary>The configuration key of the end-of-sequence token appended to the context (absent: none).</summary>
    public const string EndTokenKey = "endToken";

    /// <summary>The configuration key of the default sample count of a token model.</summary>
    public const string SampleCountKey = "sampleCount";

    /// <summary>The configuration key of the default sampling temperature.</summary>
    public const string TemperatureKey = "temperature";

    /// <summary>The configuration key of the default top-k cutoff (0: none).</summary>
    public const string TopKKey = "topK";
}
