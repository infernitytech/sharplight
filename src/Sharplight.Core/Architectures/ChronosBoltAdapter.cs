using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// The Chronos-Bolt time-series forecaster (canonical arch <c>chronos_bolt</c>, kind <see cref="ModelKinds.TimeSeriesForecasting"/>): builds
/// <see cref="TimeSeriesGraphs.ForecastGraph"/> (see <see cref="ChronosBoltGraph"/>) for the model's fixed context.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>chronos_bolt.*</c>): the T5 keys of <see cref="EncoderDecoderAdapter"/> (<c>hidden_size</c>, <c>encoder.layer_count</c>,
/// <c>decoder.layer_count</c>, <c>attention.head_count</c>, <c>attention.head_dim</c>, <c>norm_eps</c>, <c>mlp.activation</c>, <c>relative_attention.max_distance</c>)
/// plus <c>context_length</c> (observations), <c>prediction_length</c>, <c>patch_size</c> (= the patch stride), <c>quantiles</c> (comma separated levels),
/// <c>register_token</c> (default true) and <c>patch.activation</c> (default <c>mlp.activation</c>). Weights: the T5 encoder-decoder vocabulary of
/// <see cref="EncoderDecoderAdapter"/> (<c>token_embedding.weight</c> [2, H]: row 0 the decoder start, row 1 the register token) and the residual blocks
/// <c>{input,output}_patch_embedding.{hidden,output,residual}_layer.{weight,bias}</c>.
/// </remarks>
public sealed class ChronosBoltAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Chronos-Bolt forecaster";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [ChronosBoltParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = ChronosBoltParameters.From(src);
        var description = new ModelDescription
        {
            Kind = ModelKinds.TimeSeriesForecasting,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [TimeSeriesGraphs.ForecastGraph],
                    WeightPrefixes = [""],
                    Config = ComponentConfig.Of((TimeSeriesGraphs.FormKey, TimeSeriesGraphs.PatchQuantileForm), (TimeSeriesGraphs.ContextLengthKey, p.ContextLength),
                        (TimeSeriesGraphs.PredictionLengthKey, p.PredictionLength), (TimeSeriesGraphs.PatchSizeKey, p.PatchSize),
                        (TimeSeriesGraphs.QuantilesKey, string.Join(',', p.Quantiles.Select(q => q.ToString(CultureInfo.InvariantCulture))))),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal) { [TimeSeriesGraphs.ForecastGraph] = ChronosBoltGraph.Build(p, src.Weight) },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
