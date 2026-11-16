using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a CLAP model (canonical architecture <see cref="Architecture"/>, see <see cref="ClapAdapter"/>).</summary>
/// <param name="SampleRate">Sample rate of the audio.</param>
/// <param name="ClipSamples">Samples of one clip.</param>
/// <param name="FftSize">FFT size of the log-mel front end.</param>
/// <param name="HopLength">Hop of the log-mel front end.</param>
/// <param name="MelBins">Mel bins.</param>
/// <param name="SpecSize">Side of the square spectrogram image (frequency bins × time steps after the reshape).</param>
/// <param name="PatchSize">Side of a patch.</param>
/// <param name="WindowSize">Side of an attention window.</param>
/// <param name="EmbedDim">Channels after the patch embedding (doubled by every stage).</param>
/// <param name="Depths">Blocks per stage.</param>
/// <param name="Heads">Attention heads per stage.</param>
/// <param name="MlpRatio">Hidden size of the MLPs relative to the stage's channels.</param>
/// <param name="AudioEps">LayerNorm epsilon of the audio tower.</param>
/// <param name="TextLayers">Layers of the text tower.</param>
/// <param name="TextHidden">Hidden size of the text tower.</param>
/// <param name="TextHeads">Attention heads of the text tower.</param>
/// <param name="TextEps">LayerNorm epsilon of the text tower.</param>
/// <param name="TextMaxTokens">The most tokens of a text.</param>
/// <param name="PositionOffset">The first position id (padding index + 1).</param>
/// <param name="Dimensions">The embedding length.</param>
internal sealed record ClapParameters(int SampleRate, int ClipSamples, int FftSize, int HopLength, int MelBins, int SpecSize, int PatchSize, int WindowSize, int EmbedDim, int[] Depths,
    int[] Heads, int MlpRatio, float AudioEps, int TextLayers, int TextHidden, int TextHeads, float TextEps, int TextMaxTokens, int PositionOffset, int Dimensions)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "clap";

    /// <summary>The feature frames of one clip (centered frames).</summary>
    public int Frames => (ClipSamples / HopLength) + 1;

    /// <summary>The ratio of the square image's frequency side to the mel bins.</summary>
    public int FrequencyRatio => SpecSize / MelBins;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static ClapParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<int>("clip_samples"), source.Arch<int>("fft_size"), source.Arch<int>("hop_length"), source.Arch<int>("mel_bins"),
        source.Arch<int>("audio.spec_size"), source.Arch<int>("audio.patch_size"), source.Arch<int>("audio.window_size"), source.Arch<int>("audio.embed_dim"),
        source.Arch<int[]>("audio.depths"), source.Arch<int[]>("audio.head_counts"), source.Arch("audio.mlp_ratio", 4), source.Arch("audio.norm_eps", 1e-5f),
        source.Arch<int>("text.layer_count"), source.Arch<int>("text.hidden_size"), source.Arch<int>("text.attention.head_count"), source.Arch("text.norm_eps", 1e-12f),
        source.Arch<int>("text.context_length"), source.Arch<int>("text.position_offset"), source.Arch<int>("embedding_length"));
}
