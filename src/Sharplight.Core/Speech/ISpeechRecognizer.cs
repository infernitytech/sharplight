using Sharplight.Audio;

namespace Sharplight.Speech;

/// <summary>
/// Speech recognition (speech to text): audio in, text with timestamps out. Created from a loaded
/// <see cref="Tasks.TaskModel"/> (<c>model.CreateSpeechRecognizer()</c>); owns its compiled graphs and device state, not the model.
/// Not thread-safe: use one recognizer per concurrent stream.
/// </summary>
public interface ISpeechRecognizer : IDisposable
{
    /// <summary>The sample rate the model consumes (e.g. 16000); other rates are resampled.</summary>
    public int SampleRate { get; }

    /// <summary>
    /// Recognizes a recording, yielding segments as soon as they are final (long recordings are processed window by window).
    /// </summary>
    /// <param name="audio">The audio, any rate and channel count.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops recognition between decoding steps.</param>
    public IEnumerable<TranscriptionSegment> TranscribeSegments(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Recognizes a whole recording.</summary>
    /// <param name="audio">The audio, any rate and channel count.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops recognition between decoding steps.</param>
    public Transcription Transcribe(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default);
}
