namespace Sharplight.TimeSeries;

/// <summary>
/// The graph contracts of the <see cref="Models.ModelKinds.TimeSeriesForecasting"/> kind. These forms exist:
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
/// <item><description>Masked-encoder models (Moirai 1.x): one <see cref="Models.ModelComponentRoles.Encoder"/> component with one graph per patch size
/// (<see cref="MaskedEncoderGraph"/>, sizes <see cref="PatchSizesKey"/>), a bidirectional transformer over the context patches followed by the patches of
/// the horizon, which hold a learned mask embedding (<see cref="PredictMaskInput"/>). The host standardizes the series (mean and sample deviation of the
/// observed context values, see <c>MoiraiPatches.Scale</c>), left-pads the context to whole patches and returns, per horizon step, the raw parameters of a
/// mixture distribution (<see cref="ComponentsKey"/>) from which it samples.</description></item>
/// <item><description>Quantile-decoder models (Moirai 2.0): one <see cref="Models.ModelComponentRoles.Decoder"/> component with the graph
/// <see cref="QuantileDecoderGraph"/>, a causal transformer over patches with the observation mask in the input; each row predicts the
/// <see cref="PredictTokensKey"/> patches after it at every quantile level. Longer horizons feed every quantile forecast back as its own sequence; the
/// sequences run packed in one pass (<see cref="SpansInput"/>), and the quantiles of all level-by-level forecasts give the next block.</description></item>
/// <item><description>Group-encoder models (Chronos-2): one <see cref="Models.ModelComponentRoles.Encoder"/> component with the graph <see cref="GroupEncoderGraph"/>,
/// a bidirectional transformer whose layers alternate attention along time (within each series) and attention across the series of a group (at each
/// position). Every series of a group (targets, then covariates) is one sequence: its context patches, a register token row and the horizon patches, which hold
/// the known future values of covariates. The host standardizes every series with its own context (and passes it through arcsinh, <see cref="ArcsinhKey"/>),
/// packs all sequences into the rows of one pass and reads the quantile forecast of every horizon patch.</description></item>
/// </list>
/// The configuration keys sit on the encoder component (the decoder component for patch-decoder and quantile-decoder models).
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

    /// <summary>The symbol of the patch rows of <see cref="PatchDecoderGraph"/>, <see cref="MaskedEncoderGraph"/>, <see cref="QuantileDecoderGraph"/> and <see cref="GroupEncoderGraph"/>.</summary>
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

    /// <summary>
    /// The masked-encoder graph of patch size p (<see cref="MaskedEncoderGraph"/>(p)): <see cref="PatchesInput"/> (<see cref="PatchSymbol"/> rows of p scaled
    /// values; missing values and the left padding are 0 before scaling, so −loc / scale after it), <see cref="PredictMaskInput"/>, <see cref="PositionsInput"/>
    /// (the time ids), <see cref="SpansInput"/> (every row its sequence) and <see cref="ReadRowsInput"/> in, <see cref="DistributionOutput"/> out.
    /// </summary>
    public static string MaskedEncoderGraph(int patchSize) => $"masked_encoder_p{patchSize}";

    /// <summary>
    /// The quantile-decoder graph: <see cref="PatchesInput"/> (rows of p scaled values, then the observation mask; missing values and padding are 0 before
    /// scaling), <see cref="PositionsInput"/> (the time ids), <see cref="SpansInput"/> (row r sees the rows of its sequence whose time id is at most its own)
    /// and <see cref="ReadRowsInput"/> in, <see cref="TokenQuantilesOutput"/> out.
    /// </summary>
    public const string QuantileDecoderGraph = "quantile_decoder";

    /// <summary>
    /// The horizon mask of <see cref="MaskedEncoderGraph"/>, f32 <c>[patches, 1]</c>: 1 for the patches of the horizon (their embedding is replaced by the
    /// learned mask embedding), 0 for the context.
    /// </summary>
    public const string PredictMaskInput = "predict_mask";

    /// <summary>The rows whose prediction <see cref="MaskedEncoderGraph"/>, <see cref="QuantileDecoderGraph"/> and <see cref="GroupEncoderGraph"/> return, i32 <c>[<see cref="ReadSymbol"/>]</c>.</summary>
    public const string ReadRowsInput = "read_rows";

    /// <summary>The symbol of <see cref="ReadRowsInput"/>.</summary>
    public const string ReadSymbol = "R";

    /// <summary>
    /// The scaled forecast of <see cref="QuantileDecoderGraph"/>, f32 <c>[<see cref="ReadSymbol"/>, tokens · levels · p]</c>: per read row, the
    /// <see cref="PredictTokensKey"/> patches after it, each as every level of <see cref="QuantilesKey"/> × p steps.
    /// </summary>
    public const string TokenQuantilesOutput = "token_quantiles";

    /// <summary>
    /// The distribution parameters of <see cref="MaskedEncoderGraph"/>, f32 <c>[<see cref="ReadSymbol"/>, p · columns]</c>: per read row and step of its patch,
    /// the raw (unconstrained) parameters: the mixture logits (one per component), then the parameters of every component of <see cref="ComponentsKey"/>
    /// (<c>student_t</c>: df, loc, scale; <c>normal</c>: loc, scale; <c>normal_fixed:&lt;scale&gt;</c>: loc; <c>negative_binomial</c>: total count, logits;
    /// <c>log_normal</c>: loc, scale). The host applies uni2ts' domain maps (softplus for scales, counts and 2 + softplus for df).
    /// </summary>
    public const string DistributionOutput = "distribution";

    /// <summary>
    /// The group-encoder graph: <see cref="PatchesInput"/> (<see cref="PatchSymbol"/> rows of 3 · p values: the time encoding, the scaled values (0 where
    /// missing) and the observation mask; register rows are ignored), <see cref="RegisterRowsInput"/>, <see cref="KeyBiasInput"/>, <see cref="PositionsInput"/>
    /// (the row within its sequence), <see cref="SpansInput"/> (every row its sequence), <see cref="GroupOrderInput"/>, <see cref="GroupInverseInput"/>,
    /// <see cref="GroupSpansInput"/> and <see cref="ReadRowsInput"/> in, <see cref="PatchQuantilesOutput"/> out.
    /// </summary>
    public const string GroupEncoderGraph = "group_encoder";

    /// <summary>The register-token rows of <see cref="GroupEncoderGraph"/>, f32 <c>[patches, 1]</c>: 1 where the row is its sequence's register token, else 0.</summary>
    public const string RegisterRowsInput = "register_rows";

    /// <summary>
    /// The key bias of <see cref="GroupEncoderGraph"/>, f32 <c>[patches, 1]</c>: 0 for rows that take part as keys, a large negative score (−1e9) for context
    /// patches without an observed value. Both attentions use it.
    /// </summary>
    public const string KeyBiasInput = "key_bias";

    /// <summary>
    /// The group order of <see cref="GroupEncoderGraph"/>, i32 <c>[patches]</c>: the rows rearranged so that the rows of one group at one position are
    /// adjacent (group by group, position by position, series by series).
    /// </summary>
    public const string GroupOrderInput = "group_order";

    /// <summary>The inverse permutation of <see cref="GroupOrderInput"/>, i32 <c>[patches]</c>: the position of every row in the group order.</summary>
    public const string GroupInverseInput = "group_inverse";

    /// <summary>
    /// The spans of the attention across series, i32 <c>[patches, 2]</c>, in the group order: every row sees the rows of its group at its position.
    /// </summary>
    public const string GroupSpansInput = "group_spans";

    /// <summary>
    /// The scaled forecast of <see cref="GroupEncoderGraph"/>, f32 <c>[<see cref="ReadSymbol"/>, levels · p]</c>: per read row (a horizon patch) every level of
    /// <see cref="QuantilesKey"/> × p steps, standardized (and arcsinh-transformed) like the series' context.
    /// </summary>
    public const string PatchQuantilesOutput = "patch_quantiles";

    /// <summary>The <see cref="FormKey"/> of group-encoder models.</summary>
    public const string GroupEncoderForm = "group-encoder";

    /// <summary>The configuration key of the most horizon patches one group-encoder pass predicts (longer horizons are unrolled).</summary>
    public const string MaxOutputPatchesKey = "maxOutputPatches";

    /// <summary>The configuration key of the divisor of a group-encoder model's time encoding (the step offset from the forecast start).</summary>
    public const string TimeEncodingScaleKey = "timeEncodingScale";

    /// <summary>The configuration key of the arcsinh transform of a group-encoder model (bool): standardized values pass through arcsinh, forecasts through sinh.</summary>
    public const string ArcsinhKey = "arcsinh";

    /// <summary>The configuration key of a group-encoder model's register token (bool): a register row follows every sequence's context patches.</summary>
    public const string RegisterTokenKey = "registerToken";

    /// <summary>The <see cref="FormKey"/> of masked-encoder models.</summary>
    public const string MaskedEncoderForm = "masked-encoder";

    /// <summary>The <see cref="FormKey"/> of quantile-decoder models.</summary>
    public const string QuantileDecoderForm = "quantile-decoder";

    /// <summary>The configuration key of the patch sizes of a masked-encoder model (comma separated; <see cref="PatchSizeKey"/> is the default).</summary>
    public const string PatchSizesKey = "patchSizes";

    /// <summary>The configuration key of the mixture components of a masked-encoder model (comma separated, see <see cref="DistributionOutput"/>).</summary>
    public const string ComponentsKey = "components";

    /// <summary>The configuration key of the patches one row of a quantile-decoder model predicts.</summary>
    public const string PredictTokensKey = "predictTokens";

    /// <summary>The configuration key of the longest horizon of a masked-encoder or quantile-decoder model (it bounds the patch rows of one pass).</summary>
    public const string MaxHorizonKey = "maxHorizon";

    /// <summary>
    /// The configuration key of the form: <see cref="PatchQuantileForm"/>, <see cref="TokenForm"/>, <see cref="PatchDecoderForm"/>, <see cref="MaskedEncoderForm"/>,
    /// <see cref="QuantileDecoderForm"/> or <see cref="GroupEncoderForm"/>.
    /// </summary>
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
