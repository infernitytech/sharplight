using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Chronos-Bolt forecaster (<c>chronos_bolt.*</c>, documented on <see cref="ChronosBoltAdapter"/>).</summary>
internal sealed record ChronosBoltParameters
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "chronos_bolt";

    /// <summary>The T5 encoder (its <see cref="T5TextHyperParameters.ContextLength"/> is the encoder rows: patches plus the register token).</summary>
    public required T5TextHyperParameters Encoder { get; init; }

    /// <summary>Decoder layers.</summary>
    public required int DecoderLayers { get; init; }

    /// <summary>Observations per forecast.</summary>
    public required int ContextLength { get; init; }

    /// <summary>Steps per forecast.</summary>
    public required int PredictionLength { get; init; }

    /// <summary>Values per patch.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Patches per context.</summary>
    public int Patches => (ContextLength + PatchSize - 1) / PatchSize;

    /// <summary>The quantile levels, ascending.</summary>
    public required float[] Quantiles { get; init; }

    /// <summary>Whether a register token row follows the patches.</summary>
    public required bool RegisterToken { get; init; }

    /// <summary>The activation of the input and output residual blocks.</summary>
    public required string PatchActivation { get; init; }

    /// <summary>Reads the keys of <paramref name="source"/>.</summary>
    public static ChronosBoltParameters From(ModelSource source)
    {
        int hidden = source.Arch<int>("hidden_size");
        int heads = source.Arch<int>("attention.head_count");
        int context = source.Arch<int>("context_length");
        int patch = source.Arch<int>("patch_size");
        bool register = source.Arch("register_token", true);
        string activation = source.Arch("mlp.activation", "relu");
        return new ChronosBoltParameters
        {
            Encoder = new T5TextHyperParameters
            {
                HiddenSize = hidden,
                LayerCount = source.Arch<int>("encoder.layer_count"),
                HeadCount = heads,
                HeadDim = source.Arch("attention.head_dim", hidden / heads),
                NormEps = source.Arch("norm_eps", 1e-6f),
                Activation = activation,
                MaxDistance = source.Arch("relative_attention.max_distance", 128),
                ContextLength = ((context + patch - 1) / patch) + (register ? 1 : 0),
            },
            DecoderLayers = source.Arch<int>("decoder.layer_count"),
            ContextLength = context,
            PredictionLength = source.Arch<int>("prediction_length"),
            PatchSize = patch,
            Quantiles = TimeSeries.TimeSeriesQuantiles.Parse(source.Arch<string>("quantiles")),
            RegisterToken = register,
            PatchActivation = source.Arch("patch.activation", activation),
        };
    }
}
