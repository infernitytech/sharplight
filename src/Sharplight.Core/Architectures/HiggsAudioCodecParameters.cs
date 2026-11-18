using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of the Higgs Audio v2 tokenizer (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.HuggingFace</c>).</summary>
/// <param name="SampleRate">Sample rate of the audio (24 kHz).</param>
/// <param name="SemanticSampleRate">Sample rate the semantic (HuBERT) encoder works at.</param>
/// <param name="Hidden">Width of the fused latent the quantizer sees (the acoustic and the semantic width).</param>
/// <param name="AcousticHidden">Width of the acoustic encoder's output and of the acoustic decoder's input.</param>
/// <param name="SemanticHidden">Width of the semantic features.</param>
/// <param name="EncoderHidden">Channels after the acoustic encoder's first convolution (doubled per stage).</param>
/// <param name="DecoderHidden">Channels after the acoustic decoder's first convolution (halved per stage).</param>
/// <param name="DownsamplingRatios">Stride of each acoustic encoder stage.</param>
/// <param name="UpsamplingRatios">Stride of each acoustic decoder stage.</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="CodebookDim">Width of a codebook entry.</param>
/// <param name="Codebooks">The number of codebooks of the highest bandwidth.</param>
/// <param name="SemanticDownsample">Every this many HuBERT frames one semantic frame is kept.</param>
/// <param name="ConvKernel">Kernel size of the semantic encoder's convolutions.</param>
/// <param name="UnitKernel">Kernel size of the semantic encoder's residual units.</param>
/// <param name="SemanticStrides">Stride of each semantic encoder block.</param>
/// <param name="Dilations">Dilation of each residual unit of a semantic encoder block.</param>
/// <param name="Hubert">The HuBERT encoder.</param>
internal sealed record HiggsAudioCodecParameters(int SampleRate, int SemanticSampleRate, int Hidden, int AcousticHidden, int SemanticHidden, int EncoderHidden, int DecoderHidden,
    long[] DownsamplingRatios, long[] UpsamplingRatios, int CodebookSize, int CodebookDim, int Codebooks, int SemanticDownsample, int ConvKernel, int UnitKernel,
    long[] SemanticStrides, long[] Dilations, HubertParameters Hubert)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "higgs_audio_codec";

    /// <summary>The number of audio samples one frame covers.</summary>
    public long SamplesPerFrame => DownsamplingRatios.Aggregate(1L, (a, b) => a * b);

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static HiggsAudioCodecParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<int>("semantic_sample_rate"), source.Arch<int>("hidden_size"), source.Arch<int>("acoustic.hidden_size"),
        source.Arch<int>("semantic.hidden_size"), source.Arch<int>("acoustic.encoder_hidden_size"), source.Arch<int>("acoustic.decoder_hidden_size"),
        source.Arch<long[]>("acoustic.downsampling_ratios"), source.Arch<long[]>("acoustic.upsampling_ratios"), source.Arch<int>("codebook_size"), source.Arch<int>("codebook_dim"),
        source.Arch<int>("codebook_count"), source.Arch<int>("semantic.downsample"), source.Arch<int>("semantic.kernel_size"), source.Arch<int>("semantic.unit_kernel_size"),
        source.Arch<long[]>("semantic.strides"), source.Arch<long[]>("semantic.block_dilations"),
        new HubertParameters(source.Arch<int>("hubert.hidden_size"), source.Arch<int>("hubert.layers"), source.Arch<int>("hubert.heads"), source.Arch<long[]>("hubert.conv.strides"),
            source.Arch<int>("hubert.position_conv.groups"), source.Arch("hubert.norm_eps", 1e-5f)));
}
