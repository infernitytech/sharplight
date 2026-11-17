using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The 12 Hz speech codec of Qwen3-TTS (canonical arch <c>qwen3_tts_codec</c>, kind <see cref="ModelKinds.AudioCodec"/>): a Mimi encoder that emits the first 16 codebooks (built by
/// <see cref="MimiGraph"/>) and a decoder of a split residual quantizer, a sliding-window transformer, ConvNeXt upsampling and a snake-activated vocoder
/// (<see cref="Qwen3TtsCodecGraph"/>); 24 kHz audio at 12.5 frames per second.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>qwen3_tts_codec.*</c>): the Mimi keys of <see cref="MimiAdapter"/> for the encoder, and <c>decoder.{hidden_size,latent_dim,codebook_dim,codebook_size,dim,layers,
/// heads,head_dim,norm_eps,sliding_window,rope_base,quantizers,semantic_quantizers,upsample_rates,upsampling_ratios}</c> for the decoder. Weights: the Mimi encoder's canonical names
/// (<c>encoder.layers.{n}</c>, <c>encoder_transformer</c>, <c>downsample.weight</c>, <c>quantizer.{semantic,acoustic}</c>) and the decoder's <c>decoder.quantizer.{first,rest}.{input_proj,output_proj}.weight</c>,
/// <c>decoder.quantizer.{first,rest}.layers.{q}.embed</c>, <c>decoder.pre_conv</c>, <c>decoder.pre_transformer</c>, <c>decoder.upsample.{i}.{0,1}</c> and <c>decoder.decoder.{n}</c>.
/// </remarks>
public sealed class Qwen3TtsCodecAdapter : IModelAdapter
{
    /// <summary>The largest number of frames one graph run covers (longer audio runs in segments).</summary>
    public const int MaxFrames = 250;

    /// <inheritdoc/>
    public string Name => "Qwen3-TTS 12 Hz codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Qwen3TtsCodecParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var mimi = MimiParameters.From(src);
        var decoder = Qwen3TtsCodecParameters.From(src);
        var common = new (string, object)[]
        {
            (AudioCodecGraphs.SampleRateKey, mimi.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, mimi.SamplesPerFrame), (AudioCodecGraphs.CodebooksKey, mimi.Codebooks),
            (AudioCodecGraphs.DefaultCodebooksKey, mimi.DefaultCodebooks), (AudioCodecGraphs.CodebookSizeKey, mimi.CodebookSize),
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
                    Config = ComponentConfig.Of([.. common, (AudioCodecGraphs.PositionsNumeratorKey, 1), (AudioCodecGraphs.PositionsDenominatorKey, mimi.SamplesPerConvFrame)]),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames * mimi.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = ["decoder."],
                    Config = ComponentConfig.Of([.. common, (AudioCodecGraphs.PositionsNumeratorKey, 1), (AudioCodecGraphs.PositionsDenominatorKey, 1)]),
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
                [AudioCodecGraphs.EncodeGraph] = MimiGraph.BuildEncoder(mimi, src.Weight),
                [AudioCodecGraphs.DecodeGraph] = Qwen3TtsCodecGraph.BuildDecoder(decoder, src.Weight),
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
