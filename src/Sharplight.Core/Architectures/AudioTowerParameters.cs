using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The canonical hyper-parameters of the audio tower of an audio language model. Two towers: "chunked-conv2d" (Qwen3-ASR's Whisper-derived
/// encoder): log-mel frames cut into chunks of <see cref="ChunkFrames"/> frames, each chunk subsampled by <see cref="ConvCount"/> strided 3×3
/// convolutions over (mel bins, frames) with GELU, flattened and projected, plus a sinusoid table restarted in every chunk; then pre-norm
/// transformer layers whose attention stays within windows of <see cref="WindowFrames"/> frames (<see cref="ChunkedAudioTowerGraph"/>).
/// "whisper" (Voxtral): Whisper's encoder over one fixed window of <see cref="WindowFrames"/> frames, then every <see cref="Downsample"/> rows
/// concatenated (<see cref="WhisperAudioTowerGraph"/>). Both end with a LayerNorm and a two-layer GELU projector into the language model's
/// embedding width; the rows replace the audio placeholder tokens of the prompt (<see cref="AudioPromptSettings"/>).
/// </summary>
/// <remarks>
/// Keys under <c>{arch}.audio.</c>: <c>tower</c> ("chunked-conv2d" or "whisper"; the presence of the tower), <c>layer_count</c>,
/// <c>hidden_size</c>, <c>head_count</c>, optional <c>norm_eps</c> (default 1e-5), <c>activation</c> (default "gelu"), <c>mel_bins</c>;
/// chunked: <c>chunk_frames</c> (default 100), <c>window_frames</c> (default 800); whisper: <c>window_frames</c> (default 3000),
/// <c>downsample</c> (default 4); plus the feature-extractor keys of <see cref="SpeechRecognitionDescription"/> and the prompt keys of
/// <see cref="AudioPromptSettings"/>.
/// Weights (prefix <c>audio.</c>): chunked: <c>conv.{i}.weight</c> [O, C, 3 (mel), 3 (frames)] as torch's Conv2d stores them and
/// <c>conv.{i}.bias</c>, <c>conv_out.weight</c> [hidden, O · mel bins after the convolutions] (channel-major flattening, no bias); whisper:
/// <c>conv.{i}.weight</c> [O, C, 3] with biases and <c>position_embedding.weight</c> [rows, hidden]. Then per layer <c>layers.{i}.</c>
/// <c>input_norm</c>, <c>attention.{q,k,v,o}</c> (biases optional), <c>post_attention_norm</c>, <c>mlp.{up,down}</c>; <c>output_norm</c>;
/// <c>projector.1</c> and <c>projector.2</c> (biases optional).
/// </remarks>
internal sealed record AudioTowerParameters
{
    /// <summary>The value of <c>audio.tower</c> for the chunked tower.</summary>
    public const string ChunkedConv2d = "chunked-conv2d";

    /// <summary>The value of <c>audio.tower</c> for the Whisper-encoder tower.</summary>
    public const string Whisper = "whisper";

    /// <summary>The tower type (<see cref="ChunkedConv2d"/> or <see cref="Whisper"/>).</summary>
    public required string Tower { get; init; }

    /// <summary>Consecutive rows concatenated into one before the projector (1 for none).</summary>
    public int Downsample { get; init; } = 1;

    /// <summary>The prefix of the tower's weights.</summary>
    public const string WeightPrefix = "audio.";

    /// <summary>Transformer layers.</summary>
    public required int Layers { get; init; }

    /// <summary>Width of the tower's rows.</summary>
    public required int Hidden { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>LayerNorm epsilon.</summary>
    public required float Eps { get; init; }

    /// <summary>The MLP activation.</summary>
    public required string Activation { get; init; }

    /// <summary>Log-mel frames per convolution chunk.</summary>
    public required int ChunkFrames { get; init; }

    /// <summary>Log-mel frames per attention window (a multiple of <see cref="ChunkFrames"/>).</summary>
    public required int WindowFrames { get; init; }

    /// <summary>Mel bins of the features.</summary>
    public required int MelBins { get; init; }

    /// <summary>Strided convolutions of the subsampling stem.</summary>
    public required int ConvCount { get; init; }

    /// <summary>The tower's parameters, or <see langword="null"/> when the source has no audio tower.</summary>
    /// <param name="src">The model source.</param>
    /// <exception cref="NotSupportedException">The tower type is unknown.</exception>
    public static AudioTowerParameters? From(ModelSource src)
    {
        string tower = src.Arch("audio.tower", "");
        if (tower.Length == 0)
        {
            return null;
        }

        if (tower is not (ChunkedConv2d or Whisper))
        {
            throw new NotSupportedException($"{src.Architecture}: unknown audio tower '{tower}'.");
        }

        int convs = 0;
        while (src.Weight($"{WeightPrefix}conv.{convs}.weight") is not null)
        {
            convs++;
        }

        if (convs == 0)
        {
            throw new NotSupportedException($"{src.Architecture}: the audio tower has no convolution weights ('{WeightPrefix}conv.0.weight').");
        }

        bool whisper = tower == Whisper;
        int windowFrames = src.Arch("audio.window_frames", whisper ? src.Arch("audio.chunk_samples", 480000) / src.Arch("audio.hop_length", 160) : 800);
        var p = new AudioTowerParameters
        {
            Tower = tower,
            Downsample = whisper ? src.Arch("audio.downsample", 4) : 1,
            Layers = src.Arch<int>("audio.layer_count"),
            Hidden = src.Arch<int>("audio.hidden_size"),
            Heads = src.Arch<int>("audio.head_count"),
            Eps = src.Arch("audio.norm_eps", 1e-5f),
            Activation = src.Arch("audio.activation", "gelu"),
            ChunkFrames = whisper ? windowFrames : src.Arch("audio.chunk_frames", 100),
            WindowFrames = windowFrames,
            MelBins = src.Arch<int>("audio.mel_bins"),
            ConvCount = convs,
        };
        if (p.WindowFrames % p.ChunkFrames != 0 || p.Hidden % p.Heads != 0)
        {
            throw new InvalidDataException($"{src.Architecture}: audio.window_frames must be a multiple of audio.chunk_frames and audio.hidden_size of audio.head_count.");
        }

        return p;
    }

    /// <summary>The configuration of the encoder component that the host-side layout (<see cref="Speech.ChunkedAudioLayout"/>) reads.</summary>
    public IReadOnlyDictionary<string, object> ComponentConfig() => new Dictionary<string, object>(StringComparer.Ordinal)
    {
        ["tower"] = Tower,
        ["downsample"] = Downsample,
        ["chunkFrames"] = ChunkFrames,
        ["windowFrames"] = WindowFrames,
        ["convCount"] = ConvCount,
        ["melBins"] = MelBins,
    };
}
