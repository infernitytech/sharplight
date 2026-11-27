using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Salesforce's Moirai time-series forecasters (uni2ts; canonical archs <c>moirai</c> for Moirai 1.x and <c>moirai2</c> for Moirai 2.0, kind
/// <see cref="ModelKinds.TimeSeriesForecasting"/>, see <see cref="MoiraiGraph"/>): Moirai 1.x is the masked-encoder form of <see cref="TimeSeriesGraphs"/>
/// (one <see cref="TimeSeriesGraphs.MaskedEncoderGraph"/> per patch size), Moirai 2.0 the quantile-decoder form (<see cref="TimeSeriesGraphs.QuantileDecoderGraph"/>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>moirai.*</c> or <c>moirai2.*</c>): <c>hidden_size</c>, <c>intermediate_size</c>, <c>layer_count</c>, <c>attention.head_count</c>
/// (default the width / 64), <c>norm_eps</c> (1e-5), <c>rope.base</c> (10000), <c>patch_size</c> (the default patch size), <c>context_length</c> (1000 / 1680,
/// the uni2ts notebook's), <c>prediction_length</c> (the default horizon; Moirai 2.0: the steps of one pass), <c>max_horizon</c> (2048), <c>max_sequence_length</c>
/// (informational: the training length in patches); Moirai 1.x: <c>patch_sizes</c> (comma separated), <c>components</c> (the mixture, see
/// <see cref="TimeSeriesGraphs.ComponentsKey"/>), <c>sample_count</c> (100); Moirai 2.0: <c>predict_tokens</c>, <c>quantiles</c>.
/// Weights: per layer <c>encoder.layers.{i}.norm{1,2}.weight</c>, <c>encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight</c>,
/// <c>encoder.layers.{i}.self_attn.{q,k}_norm.weight</c>, <c>encoder.layers.{i}.ffn.{fc_gate,fc1,fc2}.weight</c>, <c>encoder.norm.weight</c>; Moirai 1.x:
/// <c>mask_encoding.weight</c> [1, d], per patch size p <c>in_proj.p{p}.{weight,bias}</c> and the fused distribution head <c>param_proj.p{p}.{weight,bias}</c>
/// (rows ordered step-major, see <see cref="TimeSeriesGraphs.DistributionOutput"/>); Moirai 2.0: the residual blocks
/// <c>{in,out}_proj.{hidden,output,residual}_layer.{weight,bias}</c>.
/// </remarks>
public sealed class MoiraiAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Moirai forecaster";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [MoiraiParameters.Architecture, MoiraiParameters.Architecture2];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = MoiraiParameters.From(src);
        var graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal);
        ModelComponent component;
        if (p.Is2)
        {
            graphs[TimeSeriesGraphs.QuantileDecoderGraph] = MoiraiGraph.BuildDecoder(p, src.Weight);
            int sequence = Patches(p.ContextLength, p.PatchSize) + Patches(p.MaxHorizon, p.PatchSize) + p.PredictTokens;
            component = new ModelComponent
            {
                Role = ModelComponentRoles.Decoder,
                Graphs = [TimeSeriesGraphs.QuantileDecoderGraph],
                WeightPrefixes = [""],
                SymbolBounds = new Dictionary<string, long>
                {
                    [TimeSeriesGraphs.PatchSymbol] = p.Quantiles.Length * sequence,
                    [TimeSeriesGraphs.ReadSymbol] = p.Quantiles.Length,
                },
                Config = ComponentConfig.Of((TimeSeriesGraphs.FormKey, TimeSeriesGraphs.QuantileDecoderForm), (TimeSeriesGraphs.ContextLengthKey, p.ContextLength),
                    (TimeSeriesGraphs.PredictionLengthKey, p.PredictionLength), (TimeSeriesGraphs.PatchSizeKey, p.PatchSize),
                    (TimeSeriesGraphs.PredictTokensKey, p.PredictTokens), (TimeSeriesGraphs.MaxHorizonKey, p.MaxHorizon),
                    (TimeSeriesGraphs.QuantilesKey, string.Join(',', p.Quantiles.Select(q => q.ToString(CultureInfo.InvariantCulture))))),
            };
        }
        else
        {
            foreach (int size in p.PatchSizes)
            {
                graphs[TimeSeriesGraphs.MaskedEncoderGraph(size)] = MoiraiGraph.BuildEncoder(p, size, src.Weight);
            }

            int smallest = p.PatchSizes.Min();
            component = new ModelComponent
            {
                Role = ModelComponentRoles.Encoder,
                Graphs = [.. graphs.Keys],
                WeightPrefixes = [""],
                SymbolBounds = new Dictionary<string, long>
                {
                    [TimeSeriesGraphs.PatchSymbol] = Patches(p.ContextLength, smallest) + Patches(p.MaxHorizon, smallest),
                    [TimeSeriesGraphs.ReadSymbol] = Patches(p.MaxHorizon, smallest),
                },
                Config = ComponentConfig.Of((TimeSeriesGraphs.FormKey, TimeSeriesGraphs.MaskedEncoderForm), (TimeSeriesGraphs.ContextLengthKey, p.ContextLength),
                    (TimeSeriesGraphs.PredictionLengthKey, p.PredictionLength), (TimeSeriesGraphs.PatchSizeKey, p.PatchSize),
                    (TimeSeriesGraphs.PatchSizesKey, string.Join(',', p.PatchSizes)), (TimeSeriesGraphs.ComponentsKey, p.Components),
                    (TimeSeriesGraphs.MaxHorizonKey, p.MaxHorizon), (TimeSeriesGraphs.SampleCountKey, p.SampleCount)),
            };
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = new ModelDescription { Kind = ModelKinds.TimeSeriesForecasting, Components = [component] },
        };
    }

    private static int Patches(int steps, int patchSize) => (steps + patchSize - 1) / patchSize;
}
