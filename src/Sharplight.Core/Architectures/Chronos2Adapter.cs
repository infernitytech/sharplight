using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Amazon's Chronos-2 forecaster (canonical arch <c>chronos2</c>, kind <see cref="ModelKinds.TimeSeriesForecasting"/>): the group-encoder form of
/// <see cref="TimeSeriesGraphs"/> (<see cref="TimeSeriesGraphs.GroupEncoderGraph"/>, see <see cref="Chronos2Graph"/>), which forecasts the targets of a group
/// jointly with past-only and known-future covariates.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>chronos2.*</c>): <c>hidden_size</c>, <c>intermediate_size</c>, <c>layer_count</c>, <c>attention.head_count</c>,
/// <c>attention.head_dim</c> (default the width / heads), <c>norm_eps</c> (1e-6), <c>rope.base</c> (10000), <c>mlp.activation</c> (relu; also the residual
/// blocks'), <c>context_length</c>, <c>patch_size</c> (input = output patch = stride), <c>max_output_patches</c> (64), <c>quantiles</c> (comma separated),
/// <c>register_token</c> (true), <c>time_encoding_scale</c> (default the context length) and <c>arcsinh</c> (false).
/// Weights: <c>token_embedding.weight</c> [2, H] (row 1 the register token), the residual blocks <c>{input,output}_patch_embedding.{hidden,output,residual}_layer.{weight,bias}</c>
/// (input 3 · patch → H, output H → levels · patch), per layer <c>layers.{i}.{time,group}_attention_norm.weight</c>,
/// <c>layers.{i}.{time,group}_attention.{q,k,v,o}.weight</c>, <c>layers.{i}.mlp_norm.weight</c>, <c>layers.{i}.mlp.{up,down}.weight</c>, and
/// <c>output_norm.weight</c>.
/// </remarks>
public sealed class Chronos2Adapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Chronos-2 forecaster";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Chronos2Parameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = Chronos2Parameters.From(src);
        var description = new ModelDescription
        {
            Kind = ModelKinds.TimeSeriesForecasting,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [TimeSeriesGraphs.GroupEncoderGraph],
                    WeightPrefixes = [""],
                    // One series at the longest context and horizon; the client recompiles with larger bounds for bigger groups.
                    SymbolBounds = new Dictionary<string, long>
                    {
                        [TimeSeriesGraphs.PatchSymbol] = p.SequenceRows,
                        [TimeSeriesGraphs.ReadSymbol] = p.MaxOutputPatches,
                    },
                    Config = ComponentConfig.Of((TimeSeriesGraphs.FormKey, TimeSeriesGraphs.GroupEncoderForm), (TimeSeriesGraphs.ContextLengthKey, p.ContextLength),
                        (TimeSeriesGraphs.PredictionLengthKey, p.MaxOutputPatches * p.PatchSize), (TimeSeriesGraphs.PatchSizeKey, p.PatchSize),
                        (TimeSeriesGraphs.MaxOutputPatchesKey, p.MaxOutputPatches), (TimeSeriesGraphs.TimeEncodingScaleKey, p.TimeEncodingScale),
                        (TimeSeriesGraphs.ArcsinhKey, p.Arcsinh), (TimeSeriesGraphs.RegisterTokenKey, p.RegisterToken),
                        (TimeSeriesGraphs.QuantilesKey, string.Join(',', p.Quantiles.Select(q => q.ToString(CultureInfo.InvariantCulture))))),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal) { [TimeSeriesGraphs.GroupEncoderGraph] = Chronos2Graph.Build(p, src.Weight) },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
