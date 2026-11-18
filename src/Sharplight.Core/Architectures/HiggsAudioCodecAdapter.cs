using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The Higgs Audio v2 tokenizer (canonical arch <c>higgs_audio_codec</c>, kind <see cref="ModelKinds.AudioCodec"/>): 24 kHz audio at 25 frames per second as 8 codebooks of 1024
/// entries, from a DAC-style acoustic encoder fused with layer-averaged HuBERT features (<see cref="HiggsAudioCodecGraph"/>). Builds <see cref="AudioCodecGraphs.EncodeGraph"/>
/// and <see cref="AudioCodecGraphs.DecodeGraph"/>; it is also the vocoder of <see cref="OmniVoiceAdapter"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>higgs_audio_codec.*</c>): <c>sample_rate</c>, <c>semantic_sample_rate</c>, <c>hidden_size</c>, <c>codebook_size</c>, <c>codebook_dim</c>, <c>codebook_count</c>,
/// <c>acoustic.{hidden_size,encoder_hidden_size,decoder_hidden_size,downsampling_ratios,upsampling_ratios}</c>, <c>semantic.{hidden_size,downsample,kernel_size,unit_kernel_size,
/// strides,block_dilations}</c> and <c>hubert.{hidden_size,layers,heads,conv.strides,position_conv.groups,norm_eps}</c>. Weights: <c>acoustic_encoder.*</c> and <c>acoustic_decoder.*</c>
/// (snake parameters flattened to [C]), <c>encoder_semantic.*</c>, <c>fc.*</c>, <c>fc2.*</c>, <c>quantizer.quantizers.{q}.{codebook.embed,project_in.*,project_out.*}</c> and the HuBERT
/// as <c>semantic.encoder.*</c> (<see cref="Wav2Vec2CtcAdapter"/>'s names, weight norm folded).
/// </remarks>
public sealed class HiggsAudioCodecAdapter : IModelAdapter
{
    /// <summary>The largest number of frames one graph run covers (longer audio runs in segments).</summary>
    public const int MaxFrames = 750;

    /// <inheritdoc/>
    public string Name => "Higgs Audio v2 tokenizer";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [HiggsAudioCodecParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = HiggsAudioCodecParameters.From(src);
        var config = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, p.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, (int)p.SamplesPerFrame),
            (AudioCodecGraphs.CodebooksKey, p.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, p.Codebooks), (AudioCodecGraphs.CodebookSizeKey, p.CodebookSize));
        var kernel = SincResampleKernel.Create(p.SampleRate, p.SemanticSampleRate);
        var weights = new Dictionary<string, WeightInfo>(src.Weights, StringComparer.Ordinal) { [HiggsAudioCodecGraph.ResampleKernel] = kernel.ToWeight(HiggsAudioCodecGraph.ResampleKernel) };
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioCodec,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [AudioCodecGraphs.EncodeGraph],
                    WeightPrefixes = ["acoustic_encoder.", "encoder_semantic.", "semantic.", "fc.", "quantizer.", "resample."],
                    Config = config,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames * p.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = ["acoustic_decoder.", "fc2.", "quantizer."],
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
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
            {
                [AudioCodecGraphs.EncodeGraph] = HiggsAudioCodecGraph.BuildEncoder(p, kernel, n => weights.GetValueOrDefault(n)),
                [AudioCodecGraphs.DecodeGraph] = HiggsAudioCodecGraph.BuildDecoder(p, n => weights.GetValueOrDefault(n)),
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
