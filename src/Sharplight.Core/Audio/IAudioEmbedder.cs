using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Audio embeddings and zero-shot audio tagging (kind <see cref="Models.ModelKinds.AudioEmbedding"/>: CLAP): audio and text are embedded into one space. Follows the
/// task-client pattern of every model kind (<see cref="ITaskClient"/>). The calls run the compiled graphs off the caller's thread, one request at a time.
/// </summary>
public interface IAudioEmbedder : ITaskClient
{
    /// <summary>The length of an embedding vector.</summary>
    public int Dimensions { get; }

    /// <summary>The sample rate of the audio the model takes (other audio is resampled).</summary>
    public int SampleRate { get; }

    /// <summary>The duration of one clip the audio encoder takes (longer audio is cut or averaged, shorter audio is repeated and padded).</summary>
    public TimeSpan ClipDuration { get; }

    /// <summary>Embeds audio (converted to mono at <see cref="SampleRate"/>).</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<float[]> EmbedAudioAsync(AudioBuffer audio, AudioEmbeddingOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Embeds a text into the audio embedding space.</summary>
    /// <param name="text">The text (a caption or a class prompt such as "the sound of a dog barking").</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<float[]> EmbedTextAsync(string text, AudioEmbeddingOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Scores audio against candidate texts (zero-shot audio tagging).</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="candidates">The class prompts, used as labels of the result.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<AudioClassification> ClassifyZeroShotAsync(AudioBuffer audio, IReadOnlyList<string> candidates, AudioClassificationOptions? options = null,
        CancellationToken cancellationToken = default);
}
