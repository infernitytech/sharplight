namespace Sharplight.Audio;

/// <summary>Per-request settings of <see cref="IAudioEmbedder.ClassifyZeroShotAsync"/>.</summary>
public sealed class AudioClassificationOptions
{
    /// <summary>The number of classes returned (0 for all of them).</summary>
    public int TopK { get; set; }

    /// <summary>What happens to audio longer than the model's clip.</summary>
    public LongAudioMode LongAudio { get; set; } = LongAudioMode.First;
}
