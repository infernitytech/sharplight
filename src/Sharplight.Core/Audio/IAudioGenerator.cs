using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Text-to-audio generation with a latent diffusion model (kind <see cref="Models.ModelKinds.AudioGeneration"/>: ACE-Step, Stable Audio Open, Stable Audio 3): a prompt
/// (and lyrics) in, music or sound out. Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>) and the request / options / updates shape of
/// image and video generation. Generation runs off the caller's thread, one request at a time per client.
/// </summary>
public interface IAudioGenerator : ITaskClient
{
    /// <summary>The model's own defaults (steps, guidance) and what it supports, which the <see langword="null"/> values of the per-request options resolve to.</summary>
    public AudioGenerationDefaults Defaults { get; }

    /// <summary>The sample rate of the generated audio.</summary>
    public int SampleRate { get; }

    /// <summary>The channels of the generated audio.</summary>
    public int Channels { get; }

    /// <summary>The latent frames per second the diffusion model works on.</summary>
    public double LatentFrameRate { get; }

    /// <summary>The values per latent frame.</summary>
    public int LatentChannels { get; }

    /// <summary>The longest audio the model generates in one request.</summary>
    public TimeSpan MaxDuration { get; }

    /// <summary>Denoises the latents of the audio (the autoencoder's decoder not applied).</summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Steps, guidance and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>The denoised latents.</returns>
    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Generates the audio.</summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Steps, guidance and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>The audio and the seed it was drawn from.</returns>
    public Task<AudioGenerationResult> GenerateAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Generates the audio, reporting each denoising step; the last update carries the finished audio.</summary>
    /// <param name="request">What to generate.</param>
    /// <param name="options">Steps, guidance and seed, or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <returns>An update per step, then one with the result.</returns>
    public IAsyncEnumerable<AudioGenerationUpdate> GenerateStreamingAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
