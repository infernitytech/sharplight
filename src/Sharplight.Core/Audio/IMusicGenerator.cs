using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Text-to-music generation (kind <see cref="Models.ModelKinds.MusicGeneration"/>: MusicGen): a prompt in, audio out. Follows the task-client pattern of every model
/// kind (<see cref="ITaskClient"/>) and the request / options / updates shape of the generation kinds. Generation runs off the caller's thread, one request at a time
/// per client.
/// </summary>
public interface IMusicGenerator : ITaskClient
{
    /// <summary>The model's own defaults (guidance, sampling), which the <see langword="null"/> values of the per-request options resolve to.</summary>
    public MusicGenerationDefaults Defaults { get; }

    /// <summary>The sample rate of the generated audio.</summary>
    public int SampleRate { get; }

    /// <summary>The code frames per second.</summary>
    public double FrameRate { get; }

    /// <summary>The longest music the model generates in one request.</summary>
    public TimeSpan MaxDuration { get; }

    /// <summary>Generates the codec codes of the music (the codebook streams of the model's audio codec, the delay pattern removed).</summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Guidance, sampling and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>The codes.</returns>
    public Task<CodecCodes> GenerateCodesAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Generates the music.</summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Guidance, sampling and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>The audio and the seed of its sampling.</returns>
    public Task<AudioGenerationResult> GenerateAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates the music, reporting the generated code frames and handing out the audio while it is being generated (<see cref="AudioGenerationUpdate.Audio"/>:
    /// the codes are decoded in windows with some context on both sides, so the concatenated pieces differ slightly from the result at the window borders); the
    /// last update carries the whole audio, decoded at once as <see cref="GenerateAsync"/> does.
    /// </summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Guidance, sampling and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>Updates while generating, then one with the result.</returns>
    public IAsyncEnumerable<AudioGenerationUpdate> GenerateStreamingAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
