using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The Mimi neural audio codec of Moshi (canonical arch <c>mimi</c>, kind <see cref="ModelKinds.AudioCodec"/>): a causal SEANet encoder and decoder around sliding-window
/// transformers, a 2x latent resampling and a split semantic + acoustic residual vector quantizer; 24 kHz audio at 12.5 frames per second. Builds
/// <see cref="AudioCodecGraphs.EncodeGraph"/> and <see cref="AudioCodecGraphs.DecodeGraph"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>mimi.*</c>): <c>sample_rate</c>, <c>frame_rate</c>, <c>hidden_size</c>, <c>num_filters</c>, <c>kernel_size</c>, <c>last_kernel_size</c>,
/// <c>residual_kernel_size</c>, <c>dilation_growth_rate</c>, <c>residual_layers</c>, <c>compress</c>, <c>upsampling_ratios</c> (long[]), <c>codebook_size</c>, <c>codebook_dim</c>,
/// <c>semantic_codebooks</c>, <c>codebook_count</c>, <c>default_codebooks</c>, <c>transformer.{layers,heads,head_dim,norm_eps,sliding_window,activation,rope_base}</c>,
/// <c>upsample_groups</c>. Weights: <c>encoder.layers.{n}</c> and <c>decoder.layers.{n}</c> (<c>weight</c>, <c>bias</c>; residual units <c>.block.1</c>, <c>.block.3</c>),
/// <c>{encoder,decoder}_transformer.layers.{i}</c> (<c>input_layernorm</c>, <c>self_attn.{q,k,v,o}_proj.weight</c>, <c>self_attn_layer_scale.scale</c>,
/// <c>post_attention_layernorm</c>, <c>mlp.fc1.weight</c>, <c>mlp.fc2.weight</c>, <c>mlp_layer_scale.scale</c>), <c>downsample.weight</c>, <c>upsample.weight</c>,
/// <c>quantizer.{semantic,acoustic}.{input_proj,output_proj}.weight</c> and <c>quantizer.{semantic,acoustic}.layers.{q}.embed</c> [size, dim].
/// </remarks>
public sealed class MimiAdapter : IModelAdapter
{
    /// <summary>The largest number of frames one graph run covers (longer audio runs in segments).</summary>
    public const int MaxFrames = 250;

    /// <inheritdoc/>
    public string Name => "Mimi audio codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [MimiParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = MimiParameters.From(src);
        var common = new (string, object)[]
        {
            (AudioCodecGraphs.SampleRateKey, p.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioCodecGraphs.CodebooksKey, p.Codebooks),
            (AudioCodecGraphs.DefaultCodebooksKey, p.DefaultCodebooks), (AudioCodecGraphs.CodebookSizeKey, p.CodebookSize),
        };
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioCodec,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [AudioCodecGraphs.EncodeGraph],
                    WeightPrefixes = ["encoder", "downsample.", "quantizer."],
                    Config = ComponentConfig.Of([.. common, (AudioCodecGraphs.PositionsNumeratorKey, 1), (AudioCodecGraphs.PositionsDenominatorKey, p.SamplesPerConvFrame)]),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames * p.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = ["decoder", "upsample.", "quantizer."],
                    Config = ComponentConfig.Of([.. common, (AudioCodecGraphs.PositionsNumeratorKey, (int)p.LatentStride), (AudioCodecGraphs.PositionsDenominatorKey, 1)]),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames },
                },
            ],
            Config = ComponentConfig.Of(common),
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
            {
                [AudioCodecGraphs.EncodeGraph] = MimiGraph.BuildEncoder(p, src.Weight),
                [AudioCodecGraphs.DecodeGraph] = MimiGraph.BuildDecoder(p, src.Weight),
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
