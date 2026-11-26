namespace Sharplight.TimeSeries;

/// <summary>
/// The graph contracts of the <see cref="Models.ModelKinds.TimeSeriesForecasting"/> kind. Three forms exist:
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
/// <item><description>Patch-decoder models (TimesFM): one <see cref="Models.ModelComponentRoles.Decoder"/> component with the graph <see cref="PatchDecoderGraph"/>,
/// a causal transformer over the patches of the series (<see cref="PatchSizeKey"/> values each, aligned to its end, so only the first patch may be
/// left-padded). The host fills missing values (leading ones dropped, the others linearly interpolated, as the timesfm package does), normalizes every
/// patch (<see cref="NormalizationKey"/>), and decodes autoregressively: the graph returns the normalized forecast of the patch after the last one,
/// <see cref="PredictionLengthKey"/> steps of the point forecast and every quantile level, which the host de-normalizes; for longer horizons it appends
/// column <see cref="DecodeIndexKey"/> to the series and runs the graph again. The patch rows are symbolic (<see cref="PatchSymbol"/>).</description></item>
/// </list>
/// The configuration keys sit on the encoder component (the decoder component for patch-decoder models).
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

    /// <summary>
    /// The patch-decoder graph: <see cref="PatchesInput"/> (<see cref="PatchSymbol"/> rows; the observation mask marks only the left padding of the
    /// first patch), <see cref="SpansInput"/>, <see cref="LastRowInput"/> and, for rotary models, <see cref="PositionsInput"/> in, <see cref="PointOutput"/> and, for models with a
    /// continuous quantile head (<see cref="QuantileHorizonKey"/> &gt; 0), <see cref="QuantileSpreadOutput"/> out.
    /// </summary>
    public const string PatchDecoderGraph = "patch_decoder";

    /// <summary>The symbol of the patch rows of <see cref="PatchDecoderGraph"/>.</summary>
    public const string PatchSymbol = "P";

    /// <summary>The patch positions of <see cref="PatchDecoderGraph"/>, i32 <c>[patches, 1]</c>: 0, 1, ... (rotary models only).</summary>
    public const string PositionsInput = "positions";

    /// <summary>The attention spans of <see cref="PatchDecoderGraph"/>, i32 <c>[patches, 2]</c>: every row <c>[0, patches]</c> (the attention is causal within them).</summary>
    public const string SpansInput = "spans";

    /// <summary>The row whose forecast <see cref="PatchDecoderGraph"/> returns, i32 <c>[1]</c>: the last patch.</summary>
    public const string LastRowInput = "last_row";

    /// <summary>
    /// The forecast of <see cref="PatchDecoderGraph"/>, f32 <c>[prediction length, 1 + quantile levels]</c>, normalized with the statistics of the last
    /// patch: per step the point forecast (the mean head), then every level of <see cref="QuantilesKey"/>.
    /// </summary>
    public const string PointOutput = "point";

    /// <summary>The continuous quantile head of <see cref="PatchDecoderGraph"/>, f32 <c>[<see cref="QuantileHorizonKey"/>, 1 + quantile levels]</c>, laid out as <see cref="PointOutput"/>.</summary>
    public const string QuantileSpreadOutput = "quantile_spread";

    /// <summary>The configuration key of the form: <see cref="PatchQuantileForm"/>, <see cref="TokenForm"/> or <see cref="PatchDecoderForm"/>.</summary>
    public const string FormKey = "form";

    /// <summary>The <see cref="FormKey"/> of patch-quantile models.</summary>
    public const string PatchQuantileForm = "patch-quantile";

    /// <summary>The <see cref="FormKey"/> of token models.</summary>
    public const string TokenForm = "tokens";

    /// <summary>The <see cref="FormKey"/> of patch-decoder models.</summary>
    public const string PatchDecoderForm = "patch-decoder";

    /// <summary>
    /// The configuration key of the patch normalization of a patch-decoder model: <see cref="FirstPatchNormalization"/> (TimesFM 1 / 2) or
    /// <see cref="RunningNormalization"/> (TimesFM 2.5).
    /// </summary>
    public const string NormalizationKey = "normalization";

    /// <summary>
    /// <see cref="NormalizationKey"/>: every patch is scaled with the mean and standard deviation of the first patch with at least three observed values
    /// (the last patch when there is none; the deviation at least 1e-6).
    /// </summary>
    public const string FirstPatchNormalization = "first-patch";

    /// <summary><see cref="NormalizationKey"/>: patch i is scaled with the running mean and standard deviation of the observed values of patches 0 ... i.</summary>
    public const string RunningNormalization = "running";

    /// <summary>The configuration key of the column of <see cref="PointOutput"/> that is the point forecast and is fed back for longer horizons (0: the mean head).</summary>
    public const string DecodeIndexKey = "decodeIndex";

    /// <summary>The configuration key of the steps of <see cref="QuantileSpreadOutput"/> (0 or absent: the model has no continuous quantile head).</summary>
    public const string QuantileHorizonKey = "quantileHorizon";

    /// <summary>
    /// The configuration key of the most observations (history and fed-back forecasts) a patch-decoder pass sees; older ones slide out. Absent:
    /// <see cref="ContextLengthKey"/>.
    /// </summary>
    public const string SequenceLimitKey = "sequenceLimit";

    /// <summary>
    /// The configuration key of TimesFM 2.5's input normalization (bool): the history is standardized by the mean and standard deviation of the context
    /// left-padded with zeros to <see cref="ContextLengthKey"/>, as the timesfm package's <c>normalize_inputs</c> at <c>max_context</c> = the context length.
    /// </summary>
    public const string NormalizeInputsKey = "normalizeInputs";

    /// <summary>The configuration key of flip invariance (bool): the forecast averages the model's forecast with the negated forecast of the negated series.</summary>
    public const string FlipInvarianceKey = "flipInvariance";

    /// <summary>The configuration key of the non-negativity rule (bool): a history without negative values gets a forecast without negative values.</summary>
    public const string NonNegativeKey = "nonNegative";

    /// <summary>The configuration key of the quantile-crossing fix (bool): levels below the median are capped by the next higher level, those above it floored by the next lower one.</summary>
    public const string FixQuantileCrossingKey = "fixQuantileCrossing";

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
