using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Moirai forecaster (<c>moirai.*</c> / <c>moirai2.*</c>, documented on <see cref="MoiraiAdapter"/>).</summary>
internal sealed record MoiraiParameters
{
    /// <summary>The canonical architecture id of Moirai 1.x (<c>Salesforce/moirai-1.1-R-*</c>).</summary>
    public const string Architecture = "moirai";

    /// <summary>The canonical architecture id of Moirai 2.0 (<c>Salesforce/moirai-2.0-R-small</c>).</summary>
    public const string Architecture2 = "moirai2";

    /// <summary>Whether this is Moirai 2.0 (the causal quantile decoder) rather than the masked encoder of Moirai 1.x.</summary>
    public required bool Is2 { get; init; }

    /// <summary>The model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>The width of the gated feed-forward layers.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Transformer layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Attention heads (uni2ts: the width / 64).</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dimensions per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>The epsilon of the RMS norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>The rotary base (rotary positions on the first half of every head, GPT-J pairs).</summary>
    public required float RopeBase { get; init; }

    /// <summary>The patch sizes (Moirai 1.x: one input and output projection each; Moirai 2.0: one).</summary>
    public required int[] PatchSizes { get; init; }

    /// <summary>The default patch size.</summary>
    public required int PatchSize { get; init; }

    /// <summary>The quantile levels (Moirai 2.0).</summary>
    public required float[] Quantiles { get; init; }

    /// <summary>The patches one row predicts (Moirai 2.0).</summary>
    public required int PredictTokens { get; init; }

    /// <summary>The mixture components (Moirai 1.x, see <see cref="TimeSeriesGraphs.ComponentsKey"/>).</summary>
    public required string Components { get; init; }

    /// <summary>The most observations of the history a forecast looks at.</summary>
    public required int ContextLength { get; init; }

    /// <summary>The default horizon (Moirai 2.0: the steps one pass predicts).</summary>
    public required int PredictionLength { get; init; }

    /// <summary>The longest horizon (bounds the patch rows of a pass).</summary>
    public required int MaxHorizon { get; init; }

    /// <summary>The default sample count of Moirai 1.x.</summary>
    public required int SampleCount { get; init; }

    /// <summary>Outputs per row of the parameter head of one step (Moirai 1.x): one logit per component, then every component's parameters.</summary>
    public int DistributionColumns
    {
        get
        {
            var components = Components.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            return components.Length + components.Sum(c => MixtureDistribution.ParameterCount(c));
        }
    }

    /// <summary>Reads the keys of <paramref name="source"/>.</summary>
    public static MoiraiParameters From(ModelSource source)
    {
        bool is2 = source.Architecture == Architecture2;
        int hidden = source.Arch<int>("hidden_size");
        int patch = source.Arch("patch_size", is2 ? 16 : 32);
        int tokens = source.Arch("predict_tokens", 1);
        return new MoiraiParameters
        {
            Is2 = is2,
            HiddenSize = hidden,
            IntermediateSize = source.Arch("intermediate_size", hidden * 4),
            LayerCount = source.Arch<int>("layer_count"),
            HeadCount = source.Arch("attention.head_count", hidden / 64),
            NormEps = source.Arch("norm_eps", 1e-5f),
            RopeBase = source.Arch("rope.base", 10000f),
            PatchSizes = is2 ? [patch] : [.. source.Arch("patch_sizes", "8,16,32,64,128").Split(',').Select(int.Parse)],
            PatchSize = patch,
            Quantiles = TimeSeriesQuantiles.Parse(source.Arch("quantiles", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")),
            PredictTokens = tokens,
            Components = source.Arch("components", "student_t,normal_fixed:0.001,negative_binomial,log_normal"),
            ContextLength = source.Arch("context_length", is2 ? 1680 : 1000),
            PredictionLength = source.Arch("prediction_length", is2 ? tokens * patch : 64),
            MaxHorizon = source.Arch("max_horizon", 2048),
            SampleCount = source.Arch("sample_count", 100),
        };
    }
}
