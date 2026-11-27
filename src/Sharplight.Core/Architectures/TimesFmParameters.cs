using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a TimesFM forecaster (<c>timesfm.*</c> / <c>timesfm2_5.*</c>, documented on <see cref="TimesFmAdapter"/>).</summary>
internal sealed record TimesFmParameters
{
    /// <summary>The canonical architecture id of TimesFM 1 / 2 (<c>google/timesfm-2.0-500m-pytorch</c>).</summary>
    public const string Architecture = "timesfm";

    /// <summary>The canonical architecture id of TimesFM 2.5 (<c>google/timesfm-2.5-200m-pytorch</c>).</summary>
    public const string Architecture25 = "timesfm2_5";

    /// <summary>Whether this is TimesFM 2.5 (rotary positions, query / key norms, sandwich norms, a continuous quantile head, running patch statistics).</summary>
    public required bool Is25 { get; init; }

    /// <summary>The model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>The width of the feed-forward layers and of the residual blocks' hidden layers.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Transformer layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dimensions per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>The epsilon of the RMS norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>Observations per forecast.</summary>
    public required int ContextLength { get; init; }

    /// <summary>The most observations one pass sees (history and fed-back forecasts).</summary>
    public required int SequenceLimit { get; init; }

    /// <summary>Steps per output patch.</summary>
    public required int PredictionLength { get; init; }

    /// <summary>Values per input patch.</summary>
    public required int PatchSize { get; init; }

    /// <summary>The quantile levels, ascending.</summary>
    public required float[] Quantiles { get; init; }

    /// <summary>The frequency embedding row added to every patch (TimesFM 1 / 2: 0 high, 1 medium, 2 low frequency).</summary>
    public required int Frequency { get; init; }

    /// <summary>The steps of the continuous quantile head (0: none).</summary>
    public required int QuantileHorizon { get; init; }

    /// <summary>The point-forecast column of the output (0: the mean head).</summary>
    public required int DecodeIndex { get; init; }

    /// <summary>The rotary base (TimesFM 2.5).</summary>
    public required float RopeBase { get; init; }

    /// <summary>TimesFM 2.5's input normalization.</summary>
    public required bool NormalizeInputs { get; init; }

    /// <summary>Flip invariance.</summary>
    public required bool FlipInvariance { get; init; }

    /// <summary>The non-negativity rule.</summary>
    public required bool NonNegative { get; init; }

    /// <summary>The quantile-crossing fix.</summary>
    public required bool FixQuantileCrossing { get; init; }

    /// <summary>Output columns per step: the point forecast, then every quantile level.</summary>
    public int Columns => Quantiles.Length + 1;

    /// <summary>Reads the keys of <paramref name="source"/>.</summary>
    public static TimesFmParameters From(ModelSource source)
    {
        bool is25 = source.Architecture == Architecture25;
        int hidden = source.Arch<int>("hidden_size");
        int heads = source.Arch<int>("attention.head_count");
        int context = source.Arch("context_length", is25 ? 16384 : 512);
        return new TimesFmParameters
        {
            Is25 = is25,
            HiddenSize = hidden,
            IntermediateSize = source.Arch("intermediate_size", hidden),
            LayerCount = source.Arch<int>("layer_count"),
            HeadCount = heads,
            HeadDim = source.Arch("attention.head_dim", hidden / heads),
            NormEps = source.Arch("norm_eps", 1e-6f),
            ContextLength = context,
            SequenceLimit = source.Arch("sequence_limit", context),
            PredictionLength = source.Arch("prediction_length", 128),
            PatchSize = source.Arch("patch_size", 32),
            Quantiles = TimeSeriesQuantiles.Parse(source.Arch("quantiles", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")),
            Frequency = source.Arch("frequency", 0),
            QuantileHorizon = source.Arch("quantile_horizon", 0),
            DecodeIndex = source.Arch("decode_index", is25 ? 5 : 0),
            RopeBase = source.Arch("rope.base", 10000f),
            NormalizeInputs = source.Arch("normalize_inputs", false),
            FlipInvariance = source.Arch("flip_invariance", false),
            NonNegative = source.Arch("non_negative", false),
            FixQuantileCrossing = source.Arch("fix_quantile_crossing", false),
        };
    }
}
