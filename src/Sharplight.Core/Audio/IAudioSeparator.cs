using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Music source separation (kind <see cref="Models.ModelKinds.AudioSeparation"/>: Demucs): a mixture in, its stems out (drums, bass, vocals, ...). Follows the
/// task-client pattern of every model kind (<see cref="ITaskClient"/>). The call runs the compiled graphs off the caller's thread, one request at a time.
/// </summary>
public interface IAudioSeparator : ITaskClient
{
    /// <summary>The sample rate of the audio the model takes and returns (other audio is resampled).</summary>
    public int SampleRate { get; }

    /// <summary>The channels of the audio the model takes and returns.</summary>
    public int Channels { get; }

    /// <summary>The names of the stems, in the order of the result.</summary>
    public IReadOnlyList<string> Stems { get; }

    /// <summary>Separates a mixture (converted to the model's rate and channel count first).</summary>
    /// <param name="mixture">The mixture.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="progress">Receives the fraction done after every segment, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Stops separation between segments.</param>
    public Task<IReadOnlyList<SeparatedStem>> SeparateAsync(AudioBuffer mixture, AudioSeparationOptions? options = null, IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);
}
