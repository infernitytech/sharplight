using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The EnCodec neural audio codec (canonical arch <c>encodec</c>, kind <see cref="ModelKinds.AudioCodec"/>): a causal SEANet encoder and decoder with an LSTM bottleneck and a
/// residual vector quantizer. Builds <see cref="AudioCodecGraphs.EncodeGraph"/> and <see cref="AudioCodecGraphs.DecodeGraph"/> over time-major signals.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>encodec.*</c>): <c>sample_rate</c>, <c>hidden_size</c>, <c>num_filters</c>, <c>kernel_size</c>, <c>last_kernel_size</c>, <c>residual_kernel_size</c>,
/// <c>dilation_growth_rate</c>, <c>residual_layers</c>, <c>compress</c>, <c>lstm_layers</c>, <c>upsampling_ratios</c> (long[]), <c>codebook_size</c>, <c>codebook_dim</c>,
/// <c>codebook_count</c>, <c>default_codebooks</c>, <c>conv_shortcut</c>. Weights: <c>encoder.layers.{n}</c> and <c>decoder.layers.{n}</c> with <c>weight</c> and <c>bias</c> for
/// convolutions (residual units: <c>.block.1</c>, <c>.block.3</c>, <c>.shortcut</c>; weight normalization folded), <c>.lstm.weight_ih_l{k}</c>, <c>.lstm.weight_hh_l{k}</c> and
/// <c>.lstm.bias_l{k}</c> (the two biases summed) for the bottleneck, and <c>quantizer.layers.{q}.codebook.embed</c> [size, dim].
/// </remarks>
public sealed class EncodecAdapter : IModelAdapter
{
    /// <summary>The largest number of frames one graph run covers (longer audio runs in segments).</summary>
    public const int MaxFrames = 1500;

    /// <inheritdoc/>
    public string Name => "EnCodec audio codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [EncodecParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = EncodecParameters.From(src);
        var config = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, p.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, p.SamplesPerFrame),
            (AudioCodecGraphs.CodebooksKey, p.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, p.DefaultCodebooks), (AudioCodecGraphs.CodebookSizeKey, p.CodebookSize));
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioCodec,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [AudioCodecGraphs.EncodeGraph],
                    WeightPrefixes = ["encoder.", "quantizer."],
                    Config = config,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames * p.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = ["decoder.", "quantizer."],
                    Config = config,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames },
                },
            ],
            Config = config,
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
            {
                [AudioCodecGraphs.EncodeGraph] = EncodecGraph.BuildEncoder(p, src.Weight),
                [AudioCodecGraphs.DecodeGraph] = EncodecGraph.BuildDecoder(p, src.Weight),
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
