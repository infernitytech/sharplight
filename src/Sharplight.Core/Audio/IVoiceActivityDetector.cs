using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Voice activity detection (kind <see cref="Models.ModelKinds.VoiceActivity"/>): audio in, the stretches of speech out. Follows the task-client pattern of every model kind
/// (<see cref="ITaskClient"/>). The calls run off the caller's thread. Not thread-safe: use one detector per concurrent request.
/// </summary>
public interface IVoiceActivityDetector : ITaskClient
{
    /// <summary>The sample rate the detector analyzes (audio is converted to mono at this rate first).</summary>
    public int SampleRate { get; }

    /// <summary>The number of samples one probability covers.</summary>
    public int FrameSamples { get; }

    /// <summary>The speech probability of every frame of the audio (frame <c>i</c> covers the samples <c>[i · FrameSamples, (i + 1) · FrameSamples)</c> at <see cref="SampleRate"/>).</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="cancellationToken">Stops the analysis between frames.</param>
    public Task<float[]> ComputeProbabilitiesAsync(AudioBuffer audio, CancellationToken cancellationToken = default);

    /// <summary>Finds the speech segments of the audio; their positions are in samples of the audio's own sample rate.</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops the analysis between frames.</param>
    public Task<IReadOnlyList<SpeechSegment>> DetectAsync(AudioBuffer audio, VoiceActivityOptions? options = null, CancellationToken cancellationToken = default);
}
