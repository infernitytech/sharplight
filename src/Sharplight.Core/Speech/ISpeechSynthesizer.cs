using Sharplight.Audio;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// Speech synthesis (kind <see cref="Models.ModelKinds.SpeechSynthesis"/>): text in, audio out. Follows the task-client pattern of
/// every model kind (<see cref="ITaskClient"/>). Implementations come from architecture packages (<see cref="ITaskClientFactory"/>).
/// Not thread-safe: use one synthesizer per concurrent request.
/// </summary>
public interface ISpeechSynthesizer : ITaskClient
{
    /// <summary>The sample rate of the produced audio (e.g. 24000).</summary>
    public int SampleRate { get; }

    /// <summary>The voices the model offers (names for <see cref="SpeechSynthesisOptions.Voice"/>); empty for single-voice models.</summary>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Synthesizes the whole utterance.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops synthesis between decoding steps.</param>
    public Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Synthesizes the utterance, yielding consecutive audio chunks as soon as they are decoded (playback while generating).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops synthesis between decoding steps.</param>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);
}
