using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// The original Chronos forecasters (canonical arch <c>chronos</c>, kind <see cref="ModelKinds.TimeSeriesForecasting"/>, the token form of
/// <see cref="TimeSeriesGraphs"/>): a T5 encoder-decoder over a vocabulary of value bins, built by <see cref="EncoderDecoderAdapter"/> (its <c>encode</c> and
/// <c>decode</c> graphs, the decoder's cross-attention cached per source) and described as an encoder and a decoder component with the tokenizer settings.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>chronos.*</c>): the keys of <see cref="EncoderDecoderAdapter"/> (with <c>cross_attention.cached</c>, <c>encoder.rows</c> = the context
/// plus the end token, <c>encoder.variable_rows</c> and <c>encoder.key_bias</c>), plus <c>forecast.context_length</c>, <c>forecast.prediction_length</c>, <c>forecast.tokens</c> (vocabulary),
/// <c>forecast.special_tokens</c>, <c>forecast.pad_token</c>, <c>forecast.end_token</c> (-1: none), <c>forecast.low_limit</c>, <c>forecast.high_limit</c> (the outer bin
/// centers), <c>forecast.sample_count</c>, <c>forecast.temperature</c> and <c>forecast.top_k</c> (sampling defaults). Weights: those of <see cref="EncoderDecoderAdapter"/>.
/// </remarks>
public sealed class ChronosAdapter : IModelAdapter
{
    private const string OutputRowsSymbol = "R";

    /// <inheritdoc/>
    public string Name => "Chronos forecaster";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["chronos"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var network = new EncoderDecoderAdapter().Build(sources, options);
        int rows = src.Arch<int>("encoder.rows");
        int prediction = src.Arch<int>("forecast.prediction_length");
        var config = ComponentConfig.Of(
            (TimeSeriesGraphs.FormKey, TimeSeriesGraphs.TokenForm),
            (TimeSeriesGraphs.ContextLengthKey, src.Arch<int>("forecast.context_length")),
            (TimeSeriesGraphs.PredictionLengthKey, prediction),
            (TimeSeriesGraphs.BinCountKey, src.Arch<int>("forecast.tokens")),
            (TimeSeriesGraphs.SpecialTokensKey, src.Arch("forecast.special_tokens", 2)),
            (TimeSeriesGraphs.PadTokenKey, src.Arch("forecast.pad_token", 0)),
            (TimeSeriesGraphs.EndTokenKey, src.Arch("forecast.end_token", 1)),
            (TimeSeriesGraphs.LowLimitKey, src.Arch("forecast.low_limit", -15f)),
            (TimeSeriesGraphs.HighLimitKey, src.Arch("forecast.high_limit", 15f)),
            (TimeSeriesGraphs.SampleCountKey, src.Arch("forecast.sample_count", 20)),
            (TimeSeriesGraphs.TemperatureKey, src.Arch("forecast.temperature", 1f)),
            (TimeSeriesGraphs.TopKKey, src.Arch("forecast.top_k", 50)));
        var description = new ModelDescription
        {
            Kind = ModelKinds.TimeSeriesForecasting,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [WellKnown.EncodeGraph],
                    WeightPrefixes = ["encoder.", "token_embedding."],
                    SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal) { [WellKnown.TokenSymbol] = rows },
                    Config = config,
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [WellKnown.DecodeGraph],
                    WeightPrefixes = ["decoder.", "token_embedding.", "output."],
                    SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal)
                    {
                        [WellKnown.TokenSymbol] = 1,
                        [WellKnown.ContextSymbol] = prediction + 1,
                        [WellKnown.EncoderSymbol] = rows,
                        [OutputRowsSymbol] = 1,
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = network.Architecture,
            Name = network.Name,
            Weights = network.Weights,
            Graphs = network.Graphs,
            States = network.States,
            Properties = network.Properties,
            SourceMetadata = network.SourceMetadata,
            Description = description,
            Attachments = network.Attachments,
            OwnedResources = network.OwnedResources,
        };
    }
}
