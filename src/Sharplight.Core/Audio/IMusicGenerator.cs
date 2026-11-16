using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Text-to-music generation (kind <see cref="Models.ModelKinds.MusicGeneration"/>: MusicGen): a prompt in, audio out. Follows the task-client pattern of every model
/// kind (<see cref="ITaskClient"/>). Generation runs off the caller's thread, one request at a time per client.
/// </summary>
public interface IMusicGenerator : ITaskClient
{
    /// <summary>The sample rate of the generated audio.</summary>
    public int SampleRate { get; }

    /// <summary>The code frames per second.</summary>
    public double FrameRate { get; }

    /// <summary>The longest music the model generates in one request.</summary>
    public TimeSpan MaxDuration { get; }

    /// <summary>Generates the codec codes of the music (the codebook streams of the model's audio codec, the delay pattern removed).</summary>
    /// <param name="request">The request.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public Task<CodecCodes> GenerateCodesAsync(MusicGenerationRequest request, CancellationToken cancellationToken = default);

    /// <summary>Generates the music.</summary>
    /// <param name="request">The request.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public Task<AudioBuffer> GenerateAsync(MusicGenerationRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates the music and returns it in pieces while it is being generated: the codes are decoded in windows with some context on both sides, so the
    /// concatenated pieces differ slightly from <see cref="GenerateAsync"/>'s audio at the window borders.
    /// </summary>
    /// <param name="request">The request.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public IAsyncEnumerable<AudioBuffer> GenerateStreamingAsync(MusicGenerationRequest request, CancellationToken cancellationToken = default);
}
