using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of an EnCodec model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.HuggingFace</c>).</summary>
/// <param name="SampleRate">Sample rate of the audio.</param>
/// <param name="Hidden">Width of the latent the quantizer sees.</param>
/// <param name="Filters">Channels of the first convolution (doubled after every downsampling).</param>
/// <param name="KernelSize">Kernel of the first convolution.</param>
/// <param name="LastKernelSize">Kernel of the last convolution.</param>
/// <param name="ResidualKernelSize">Kernel of the first convolution of a residual unit.</param>
/// <param name="DilationGrowth">Dilation growth of the residual units of a stage.</param>
/// <param name="ResidualLayers">Residual units per stage.</param>
/// <param name="Compress">Channel reduction inside a residual unit.</param>
/// <param name="LstmLayers">Layers of the recurrent bottleneck.</param>
/// <param name="Ratios">Upsampling ratio of each decoder stage (the encoder uses them in reverse).</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="CodebookDim">Width of a codebook entry.</param>
/// <param name="Codebooks">The number of codebooks in the checkpoint.</param>
/// <param name="DefaultCodebooks">The number of codebooks used by default.</param>
/// <param name="ConvShortcut">Whether the residual shortcut is a convolution.</param>
internal sealed record EncodecParameters(int SampleRate, int Hidden, int Filters, int KernelSize, int LastKernelSize, int ResidualKernelSize, int DilationGrowth, int ResidualLayers,
    int Compress, int LstmLayers, long[] Ratios, int CodebookSize, int CodebookDim, int Codebooks, int DefaultCodebooks, bool ConvShortcut)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "encodec";

    /// <summary>The number of audio samples one frame covers.</summary>
    public long SamplesPerFrame => Ratios.Aggregate(1L, (a, b) => a * b);

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static EncodecParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<int>("hidden_size"), source.Arch<int>("num_filters"), source.Arch<int>("kernel_size"), source.Arch<int>("last_kernel_size"),
        source.Arch<int>("residual_kernel_size"), source.Arch<int>("dilation_growth_rate"), source.Arch<int>("residual_layers"), source.Arch<int>("compress"),
        source.Arch<int>("lstm_layers"), source.Arch<long[]>("upsampling_ratios"), source.Arch<int>("codebook_size"), source.Arch<int>("codebook_dim"),
        source.Arch<int>("codebook_count"), source.Arch<int>("default_codebooks"), source.Arch("conv_shortcut", true));
}
