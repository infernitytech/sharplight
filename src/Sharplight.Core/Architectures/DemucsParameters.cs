using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a hybrid transformer Demucs (canonical architecture <see cref="Architecture"/>, see <see cref="DemucsAdapter"/>).</summary>
/// <param name="Sources">The stem names.</param>
/// <param name="SampleRate">Sample rate.</param>
/// <param name="AudioChannels">Audio channels (input and per stem).</param>
/// <param name="Channels">Channels of the first encoder layer.</param>
/// <param name="Growth">Channel growth per layer.</param>
/// <param name="Depth">Encoder and decoder layers per branch.</param>
/// <param name="Nfft">FFT size (the hop is a quarter of it).</param>
/// <param name="Kernel">Kernel of the encoder and decoder convolutions.</param>
/// <param name="Stride">Stride of the encoder and decoder convolutions.</param>
/// <param name="DconvDepth">Layers of a residual dilated branch.</param>
/// <param name="DconvCompression">Channel reduction inside a residual dilated branch.</param>
/// <param name="BottomChannels">Width of the transformer.</param>
/// <param name="FrequencyEmbedding">Weight of the frequency embedding added after the first spectral layer (0 for none).</param>
/// <param name="EmbeddingScale">Scale of the stored embedding.</param>
/// <param name="TransformerLayers">Layers of each transformer branch.</param>
/// <param name="TransformerHeads">Attention heads.</param>
/// <param name="TransformerHidden">Hidden size of the transformer's feed-forward layers.</param>
/// <param name="MaxPeriod">Denominator base of the sinusoidal position embeddings.</param>
/// <param name="PositionWeight">Weight of the position embeddings.</param>
/// <param name="SegmentSamples">Samples of the fixed segment the graph is built for.</param>
internal sealed record DemucsParameters(string[] Sources, int SampleRate, int AudioChannels, int Channels, int Growth, int Depth, int Nfft, int Kernel, int Stride, int DconvDepth,
    int DconvCompression, int BottomChannels, float FrequencyEmbedding, float EmbeddingScale, int TransformerLayers, int TransformerHeads, int TransformerHidden, double MaxPeriod,
    float PositionWeight, int SegmentSamples)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "demucs";

    /// <summary>The hop of the spectrogram.</summary>
    public int Hop => Nfft / 4;

    /// <summary>The frames of the spectrogram of one segment.</summary>
    public int Frames => (SegmentSamples + Hop - 1) / Hop;

    /// <summary>The frequency bins of the spectrogram input (the highest bin is dropped).</summary>
    public int Bins => Nfft / 2;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static DemucsParameters From(ModelSource source) => new(
        source.Arch<string>("sources").Split(','), source.Arch<int>("sample_rate"), source.Arch<int>("audio_channels"), source.Arch<int>("channels"), source.Arch<int>("growth"),
        source.Arch<int>("depth"), source.Arch<int>("nfft"), source.Arch<int>("kernel_size"), source.Arch<int>("stride"), source.Arch<int>("dconv_depth"), source.Arch<int>("dconv_compress"),
        source.Arch<int>("bottom_channels"), source.Arch("freq_emb", 0.2f), source.Arch("emb_scale", 10f), source.Arch<int>("transformer.layer_count"), source.Arch<int>("transformer.head_count"),
        source.Arch<int>("transformer.hidden_size"), source.Arch("transformer.max_period", 10000.0), source.Arch("transformer.position_weight", 1f), source.Arch<int>("segment_samples"));
}
