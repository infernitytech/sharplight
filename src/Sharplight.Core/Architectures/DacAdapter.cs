using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The Descript Audio Codec (canonical arch <c>dac</c>, kind <see cref="ModelKinds.AudioCodec"/>): a convolutional encoder and decoder with snake activations and a residual vector
/// quantizer over factorized, l2-normalized codes. Builds <see cref="AudioCodecGraphs.EncodeGraph"/> and <see cref="AudioCodecGraphs.DecodeGraph"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>dac.*</c>): <c>sample_rate</c>, <c>hidden_size</c>, <c>encoder_hidden_size</c>, <c>decoder_hidden_size</c>, <c>downsampling_ratios</c> and
/// <c>upsampling_ratios</c> (long[]), <c>codebook_size</c>, <c>codebook_dim</c>, <c>codebook_count</c>. Weights follow the Hugging Face names (<c>encoder.conv1</c>,
/// <c>encoder.block.{i}.res_unit{1..3}.{snake1.alpha,conv1,snake2.alpha,conv2}</c>, <c>encoder.block.{i}.{snake1.alpha,conv1}</c>, <c>encoder.snake1.alpha</c>,
/// <c>encoder.conv2</c>, <c>decoder.conv1</c>, <c>decoder.block.{i}.{snake1.alpha,conv_t1,res_unit..}</c>, <c>decoder.snake1.alpha</c>, <c>decoder.conv2</c>,
/// <c>quantizer.quantizers.{q}.{codebook.weight,in_proj,out_proj}</c>) with the snake parameters flattened to <c>[C]</c>.
/// </remarks>
public sealed class DacAdapter : IModelAdapter
{
    /// <summary>The largest number of frames one graph run covers (longer audio runs in segments).</summary>
    public const int MaxFrames = 860;

    /// <inheritdoc/>
    public string Name => "DAC audio codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [DacParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = DacParameters.From(src);
        var config = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, p.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, p.SamplesPerFrame),
            (AudioCodecGraphs.CodebooksKey, p.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, p.Codebooks), (AudioCodecGraphs.CodebookSizeKey, p.CodebookSize));
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
                [AudioCodecGraphs.EncodeGraph] = DacGraph.BuildEncoder(p, src.Weight),
                [AudioCodecGraphs.DecodeGraph] = DacGraph.BuildDecoder(p, src.Weight),
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
