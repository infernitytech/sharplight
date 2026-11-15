using Sharplight.Audio;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// Speech recognition (kind <see cref="Models.ModelKinds.SpeechRecognition"/>): audio in, text with timestamps out. Follows the
/// task-client pattern of every model kind (<see cref="ITaskClient"/>); corresponds to Microsoft.Extensions.AI's
/// <c>ISpeechToTextClient</c>. Implementations come from architecture packages (<see cref="ITaskClientFactory"/>). Not thread-safe:
/// use one recognizer per concurrent stream.
/// </summary>
public interface ISpeechRecognizer : ITaskClient
{
    /// <summary>The sample rate the model consumes (e.g. 16000); audio at other rates is resampled.</summary>
    public int SampleRate { get; }

    /// <summary>Recognizes a whole recording.</summary>
    /// <param name="audio">The audio, any rate and channel count.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops recognition between decoding steps.</param>
    public Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Recognizes a recording, yielding segments as soon as they are final (long recordings are processed window by window).</summary>
    /// <param name="audio">The audio, any rate and channel count.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops recognition between decoding steps.</param>
    public IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        CancellationToken cancellationToken = default);
}
