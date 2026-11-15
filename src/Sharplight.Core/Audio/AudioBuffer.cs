namespace Sharplight.Audio;

/// <summary>
/// Decoded PCM audio: 32-bit float samples, nominally in [-1, 1], interleaved by channel (frame-major: all channels of frame 0,
/// then frame 1, ...). Immutable by convention: the sample array is not copied, and consumers must not modify it.
/// </summary>
public sealed class AudioBuffer
{
    /// <summary>Creates a buffer over interleaved samples.</summary>
    /// <param name="samples">Interleaved samples; the length must be a multiple of <paramref name="channels"/>. Not copied.</param>
    /// <param name="sampleRate">Frames per second (positive).</param>
    /// <param name="channels">Channel count (positive).</param>
    /// <exception cref="ArgumentException">The rate or channel count is not positive, or the samples are not whole frames.</exception>
    public AudioBuffer(float[] samples, int sampleRate, int channels = 1)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (sampleRate <= 0 || channels <= 0)
        {
            throw new ArgumentException($"Sample rate ({sampleRate}) and channel count ({channels}) must be positive.");
        }

        if (samples.Length % channels != 0)
        {
            throw new ArgumentException($"{samples.Length} samples are not whole frames of {channels} channels.");
        }

        Samples = samples;
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <summary>Interleaved samples (<see cref="FrameCount"/> · <see cref="Channels"/> values).</summary>
    public float[] Samples { get; }

    /// <summary>Frames per second.</summary>
    public int SampleRate { get; }

    /// <summary>Channels per frame.</summary>
    public int Channels { get; }

    /// <summary>Number of frames (samples per channel).</summary>
    public int FrameCount => Samples.Length / Channels;

    /// <summary>Length of the audio.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds((double)FrameCount / SampleRate);

    /// <summary>A mono buffer: the average of the channels per frame, or this buffer when it already is mono.</summary>
    public AudioBuffer ToMono()
    {
        if (Channels == 1)
        {
            return this;
        }

        var mono = new float[FrameCount];
        float scale = 1f / Channels;
        for (int frame = 0; frame < mono.Length; frame++)
        {
            float sum = 0;
            int offset = frame * Channels;
            for (int channel = 0; channel < Channels; channel++)
            {
                sum += Samples[offset + channel];
            }

            mono[frame] = sum * scale;
        }

        return new AudioBuffer(mono, SampleRate, 1);
    }

    /// <summary>
    /// This audio at another sample rate (band-limited, see <see cref="AudioResampler"/>), or this buffer when the rate already matches.
    /// </summary>
    /// <param name="sampleRate">The target rate in frames per second.</param>
    public AudioBuffer Resample(int sampleRate)
        => sampleRate == SampleRate ? this : new AudioBuffer(AudioResampler.Resample(Samples, Channels, SampleRate, sampleRate), sampleRate, Channels);

    /// <summary>
    /// The mono signal at <paramref name="sampleRate"/> that speech and audio models consume (e.g. 16 kHz for Whisper): channels are
    /// averaged, then the signal is resampled.
    /// </summary>
    /// <param name="sampleRate">The model's input rate.</param>
    public AudioBuffer ToModelInput(int sampleRate) => ToMono().Resample(sampleRate);
}
