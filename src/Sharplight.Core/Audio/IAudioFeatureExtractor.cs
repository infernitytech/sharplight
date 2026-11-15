namespace Sharplight.Audio;

/// <summary>
/// Turns a mono signal into the frame-level features a speech or audio encoder consumes (the
/// <see cref="Models.ModelComponentRoles.FeatureExtractor"/> component of a model). Implementations are chosen from the component's
/// configuration by <see cref="AudioFeatureExtractors.Create"/>. Kept small on purpose: a graph-based extractor (a generic STFT op) can
/// replace the host implementation behind it without touching the task clients.
/// </summary>
internal interface IAudioFeatureExtractor
{
    /// <summary>The sample rate the extractor expects, in Hz.</summary>
    public int SampleRate { get; }

    /// <summary>Samples between consecutive feature frames.</summary>
    public int HopLength { get; }

    /// <summary>Feature channels per frame (e.g. mel bins).</summary>
    public int Channels { get; }

    /// <summary>
    /// The features of a whole recording, including whatever padding the model's front end appends (e.g. 30 s of silence for Whisper,
    /// so every window of the recording is complete).
    /// </summary>
    /// <param name="samples">Mono samples at <see cref="SampleRate"/>.</param>
    /// <returns>Channel-major features [<see cref="Channels"/>, frames].</returns>
    public AudioFeatures Extract(ReadOnlySpan<float> samples);
}
