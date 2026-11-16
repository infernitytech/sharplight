using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Mimi model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.HuggingFace</c>).</summary>
/// <param name="SampleRate">Sample rate of the audio.</param>
/// <param name="FrameRate">Frames per second of the codes.</param>
/// <param name="Hidden">Width of the latent and of the transformers.</param>
/// <param name="Filters">Channels of the first convolution (doubled after every downsampling).</param>
/// <param name="KernelSize">Kernel of the first convolution.</param>
/// <param name="LastKernelSize">Kernel of the last convolution.</param>
/// <param name="ResidualKernelSize">Kernel of the first convolution of a residual unit.</param>
/// <param name="DilationGrowth">Dilation growth of the residual units of a stage.</param>
/// <param name="ResidualLayers">Residual units per stage.</param>
/// <param name="Compress">Channel reduction inside a residual unit.</param>
/// <param name="Ratios">Upsampling ratio of each decoder stage (the encoder uses them in reverse).</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="CodebookDim">Width of a codebook entry.</param>
/// <param name="SemanticCodebooks">The number of codebooks of the semantic quantizer (the first codes).</param>
/// <param name="Codebooks">The total number of codebooks.</param>
/// <param name="DefaultCodebooks">The number of codebooks used by default.</param>
/// <param name="Layers">Layers of each transformer.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="Eps">The layer norm epsilon.</param>
/// <param name="Window">The sliding attention window.</param>
/// <param name="Activation">The MLP activation.</param>
/// <param name="RopeBase">The rotary base.</param>
internal sealed record MimiParameters(int SampleRate, double FrameRate, int Hidden, int Filters, int KernelSize, int LastKernelSize, int ResidualKernelSize, int DilationGrowth,
    int ResidualLayers, int Compress, long[] Ratios, int CodebookSize, int CodebookDim, int SemanticCodebooks, int Codebooks, int DefaultCodebooks, int Layers, int Heads,
    int HeadDim, float Eps, int Window, string Activation, float RopeBase)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "mimi";

    /// <summary>The number of audio samples one convolutional frame (before the latent downsampling) covers.</summary>
    public long SamplesPerConvFrame => Ratios.Aggregate(1L, (a, b) => a * b);

    /// <summary>The latent downsampling stride.</summary>
    public long LatentStride => (long)Math.Round(SampleRate / FrameRate / SamplesPerConvFrame);

    /// <summary>The number of audio samples one frame of codes covers.</summary>
    public long SamplesPerFrame => SamplesPerConvFrame * LatentStride;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static MimiParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<float>("frame_rate"), source.Arch<int>("hidden_size"), source.Arch<int>("num_filters"), source.Arch<int>("kernel_size"),
        source.Arch<int>("last_kernel_size"), source.Arch<int>("residual_kernel_size"), source.Arch<int>("dilation_growth_rate"), source.Arch<int>("residual_layers"),
        source.Arch<int>("compress"), source.Arch<long[]>("upsampling_ratios"), source.Arch<int>("codebook_size"), source.Arch<int>("codebook_dim"),
        source.Arch<int>("semantic_codebooks"), source.Arch<int>("codebook_count"), source.Arch<int>("default_codebooks"), source.Arch<int>("transformer.layers"),
        source.Arch<int>("transformer.heads"), source.Arch<int>("transformer.head_dim"), source.Arch<float>("transformer.norm_eps"), source.Arch<int>("transformer.sliding_window"),
        source.Arch<string>("transformer.activation"), source.Arch<float>("transformer.rope_base"));
}
