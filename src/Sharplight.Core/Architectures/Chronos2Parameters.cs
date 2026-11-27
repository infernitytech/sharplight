using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Chronos-2 forecaster (<c>chronos2.*</c>, documented on <see cref="Chronos2Adapter"/>).</summary>
internal sealed record Chronos2Parameters
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "chronos2";

    /// <summary>The model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>The feed-forward and residual-block width.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Encoder layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Values per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>The RMS norm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>The rotary base of the time attention.</summary>
    public required float RopeBase { get; init; }

    /// <summary>The activation of the feed-forward layers and the residual blocks.</summary>
    public required string Activation { get; init; }

    /// <summary>Observations per forecast.</summary>
    public required int ContextLength { get; init; }

    /// <summary>Values per patch (input and output).</summary>
    public required int PatchSize { get; init; }

    /// <summary>The most horizon patches one pass predicts.</summary>
    public required int MaxOutputPatches { get; init; }

    /// <summary>The quantile levels, ascending.</summary>
    public required float[] Quantiles { get; init; }

    /// <summary>Whether a register token row follows the context patches.</summary>
    public required bool RegisterToken { get; init; }

    /// <summary>The divisor of the time encoding (the step offsets from the forecast start).</summary>
    public required int TimeEncodingScale { get; init; }

    /// <summary>Whether the standardized values pass through arcsinh (and the forecasts through sinh).</summary>
    public required bool Arcsinh { get; init; }

    /// <summary>Patches per context.</summary>
    public int ContextPatches => (ContextLength + PatchSize - 1) / PatchSize;

    /// <summary>Rows of one series' sequence at the longest context and horizon.</summary>
    public int SequenceRows => ContextPatches + (RegisterToken ? 1 : 0) + MaxOutputPatches;

    /// <summary>Reads the keys of <paramref name="source"/>.</summary>
    public static Chronos2Parameters From(ModelSource source)
    {
        int hidden = source.Arch<int>("hidden_size");
        int heads = source.Arch<int>("attention.head_count");
        int context = source.Arch<int>("context_length");
        return new Chronos2Parameters
        {
            HiddenSize = hidden,
            IntermediateSize = source.Arch<int>("intermediate_size"),
            LayerCount = source.Arch<int>("layer_count"),
            HeadCount = heads,
            HeadDim = source.Arch("attention.head_dim", hidden / heads),
            NormEps = source.Arch("norm_eps", 1e-6f),
            RopeBase = source.Arch("rope.base", 10000f),
            Activation = source.Arch("mlp.activation", "relu"),
            ContextLength = context,
            PatchSize = source.Arch<int>("patch_size"),
            MaxOutputPatches = source.Arch("max_output_patches", 64),
            Quantiles = TimeSeries.TimeSeriesQuantiles.Parse(source.Arch<string>("quantiles")),
            RegisterToken = source.Arch("register_token", true),
            TimeEncodingScale = source.Arch("time_encoding_scale", context),
            Arcsinh = source.Arch("arcsinh", false),
        };
    }
}
