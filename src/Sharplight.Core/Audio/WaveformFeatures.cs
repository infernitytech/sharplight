using Sharplight.Models;

namespace Sharplight.Audio;

/// <summary>
/// The raw waveform as the features of encoders with their own convolutional front end (feature-extractor <c>type</c> <c>waveform</c>,
/// Moonshine): one channel, one frame per sample, optionally normalized to zero mean and unit variance per recording (transformers'
/// <c>Wav2Vec2FeatureExtractor</c> with <c>do_normalize</c>). Nothing is appended: such encoders take windows of any length.
/// </summary>
/// <remarks>Component configuration: <c>sampleRate</c>, optional <c>normalize</c> (default false).</remarks>
internal sealed class WaveformFeatures : IAudioFeatureExtractor
{
    /// <summary>The feature-extractor type this class implements.</summary>
    public const string Type = "waveform";

    private readonly bool _normalize;

    /// <summary>Creates the extractor from a feature-extractor component's configuration.</summary>
    /// <param name="component">The component (configuration keys in the remarks).</param>
    /// <exception cref="InvalidDataException">The sample rate is missing.</exception>
    public WaveformFeatures(ModelComponent component)
    {
        SampleRate = component.ConfigValue("sampleRate", 0);
        if (SampleRate <= 0)
        {
            throw new InvalidDataException($"waveform feature extractor: sampleRate {SampleRate} must be positive.");
        }

        _normalize = component.ConfigValue("normalize", false);
    }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public int HopLength => 1;

    /// <inheritdoc/>
    public int Channels => 1;

    /// <inheritdoc/>
    public AudioFeatures Extract(ReadOnlySpan<float> samples)
    {
        var values = samples.ToArray();
        if (_normalize && values.Length > 0)
        {
            double mean = 0;
            foreach (float v in values)
            {
                mean += v;
            }

            mean /= values.Length;
            double variance = 0;
            foreach (float v in values)
            {
                variance += (v - mean) * (v - mean);
            }

            double scale = 1 / Math.Sqrt((variance / values.Length) + 1e-7);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)((values[i] - mean) * scale);
            }
        }

        return new AudioFeatures(values, 1, values.Length);
    }
}
