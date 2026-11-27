using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Google's TimesFM time-series forecasters (canonical archs <c>timesfm</c> for TimesFM 1 / 2 and <c>timesfm2_5</c> for TimesFM 2.5, kind
/// <see cref="ModelKinds.TimeSeriesForecasting"/>): builds <see cref="TimeSeriesGraphs.PatchDecoderGraph"/> (see <see cref="TimesFmGraph"/>), the patch-decoder form.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>timesfm.*</c> or <c>timesfm2_5.*</c>): <c>hidden_size</c>, <c>intermediate_size</c>, <c>layer_count</c>, <c>attention.head_count</c>,
/// <c>attention.head_dim</c>, <c>norm_eps</c>, <c>context_length</c>, <c>sequence_limit</c> (default the context length), <c>prediction_length</c> (the output
/// patch, 128), <c>patch_size</c> (32), <c>quantiles</c> (comma separated); TimesFM 1 / 2: <c>frequency</c> (the frequency embedding row, default 0);
/// TimesFM 2.5: <c>quantile_horizon</c> (the continuous quantile head's steps), <c>decode_index</c> (5), <c>rope.base</c> and the forecasting flags
/// <c>normalize_inputs</c>, <c>flip_invariance</c>, <c>non_negative</c>, <c>fix_quantile_crossing</c>.
/// Weights of <c>timesfm</c>: <c>input_ff_layer.{input,output,residual}_layer.{weight,bias}</c>, <c>freq_emb.weight</c>, per layer
/// <c>layers.{i}.input_layernorm.weight</c>, <c>layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}</c>, <c>layers.{i}.self_attn.scaling</c> (the per-dimension
/// query scale before softplus), <c>layers.{i}.mlp.layer_norm.{weight,bias}</c>, <c>layers.{i}.mlp.{gate,down}_proj.{weight,bias}</c> and
/// <c>horizon_ff_layer.*</c>. Weights of <c>timesfm2_5</c>: <c>input_ff_layer.*</c>, per layer <c>layers.{i}.{input_layernorm,post_attention_layernorm,
/// pre_feedforward_layernorm,post_feedforward_layernorm}.weight</c>, <c>layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight</c> (or a fused
/// <c>qkv_proj.weight</c>), <c>layers.{i}.self_attn.{q_norm,k_norm}.weight</c>, <c>layers.{i}.self_attn.scaling</c>, <c>layers.{i}.mlp.{ff0,ff1}.weight</c> and the
/// bias-free heads <c>output_projection_point.*</c> and <c>output_projection_quantiles.*</c>.
/// </remarks>
public sealed class TimesFmAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "TimesFM forecaster";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [TimesFmParameters.Architecture, TimesFmParameters.Architecture25];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = TimesFmParameters.From(src);
        var description = new ModelDescription
        {
            Kind = ModelKinds.TimeSeriesForecasting,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [TimeSeriesGraphs.PatchDecoderGraph],
                    WeightPrefixes = [""],
                    SymbolBounds = new Dictionary<string, long> { [TimeSeriesGraphs.PatchSymbol] = (p.SequenceLimit + p.PatchSize - 1) / p.PatchSize },
                    Config = ComponentConfig.Of((TimeSeriesGraphs.FormKey, TimeSeriesGraphs.PatchDecoderForm), (TimeSeriesGraphs.ContextLengthKey, p.ContextLength),
                        (TimeSeriesGraphs.SequenceLimitKey, p.SequenceLimit), (TimeSeriesGraphs.PredictionLengthKey, p.PredictionLength),
                        (TimeSeriesGraphs.PatchSizeKey, p.PatchSize),
                        (TimeSeriesGraphs.QuantilesKey, string.Join(',', p.Quantiles.Select(q => q.ToString(CultureInfo.InvariantCulture)))),
                        (TimeSeriesGraphs.NormalizationKey, p.Is25 ? TimeSeriesGraphs.RunningNormalization : TimeSeriesGraphs.FirstPatchNormalization),
                        (TimeSeriesGraphs.DecodeIndexKey, p.DecodeIndex), (TimeSeriesGraphs.QuantileHorizonKey, p.QuantileHorizon),
                        (TimeSeriesGraphs.NormalizeInputsKey, p.NormalizeInputs), (TimeSeriesGraphs.FlipInvarianceKey, p.FlipInvariance),
                        (TimeSeriesGraphs.NonNegativeKey, p.NonNegative), (TimeSeriesGraphs.FixQuantileCrossingKey, p.FixQuantileCrossing)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal) { [TimeSeriesGraphs.PatchDecoderGraph] = TimesFmGraph.Build(p, src.Weight) },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
