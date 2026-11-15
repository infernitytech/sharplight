using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hyper-parameters of a WavTokenizer decoder (canonical architecture <see cref="Architecture"/>): the codebook, the PosNet and ConvNeXt
/// stacks and the geometry of the inverse STFT that turns the head's log-magnitudes and phases into a waveform.
/// </summary>
/// <param name="VocabSize">Codebook size.</param>
/// <param name="CodebookDim">Width of a codebook entry.</param>
/// <param name="HiddenSize">Channel count of the PosNet and ConvNeXt stacks.</param>
/// <param name="PosnetLayers">Number of PosNet blocks (resnet, resnet, attention, resnet, resnet, norm).</param>
/// <param name="Blocks">Number of ConvNeXt blocks.</param>
/// <param name="FeedForward">Width of the ConvNeXt pointwise expansion.</param>
/// <param name="NormEps">Epsilon of the layer norms.</param>
/// <param name="GroupNormEps">Epsilon of the group norms.</param>
/// <param name="GroupNormGroups">Group count of the group norms.</param>
/// <param name="NFft">Length of the inverse FFT; the head produces <c>NFft / 2 + 1</c> log-magnitudes and as many phases.</param>
/// <param name="HopLength">Samples between frames (the codec's frame rate is <c>SampleRate / HopLength</c>).</param>
/// <param name="SampleRate">Sample rate of the decoded audio.</param>
internal sealed record WavTokenizerParameters(int VocabSize, int CodebookDim, int HiddenSize, int PosnetLayers, int Blocks, int FeedForward,
    float NormEps, float GroupNormEps, int GroupNormGroups, int NFft, int HopLength, int SampleRate)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "wavtokenizer";

    /// <summary>The number of spectrum bins (<c>NFft / 2 + 1</c>).</summary>
    public int Bins => (NFft / 2) + 1;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static WavTokenizerParameters From(ModelSource source) => new(
        source.Arch<int>("vocab_size"), source.Arch<int>("codebook_dim"), source.Arch<int>("hidden_size"), source.Arch<int>("posnet_layers"),
        source.Arch<int>("block_count"), source.Arch<int>("feed_forward_length"), source.Arch<float>("norm_eps"), source.Arch<float>("group_norm_eps"),
        source.Arch<int>("group_norm_groups"), source.Arch<int>("n_fft"), source.Arch<int>("hop_length"), source.Arch<int>("sample_rate"));
}
