using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// Text-to-audio generation with a latent diffusion model (kind <see cref="Models.ModelKinds.AudioGeneration"/>: ACE-Step): a prompt (and lyrics) in, music or sound
/// out. Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Generation runs off the caller's thread, one request at a time per client.
/// </summary>
public interface IAudioGenerator : ITaskClient
{
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

    /// <summary>The number of denoising steps used when a request has none.</summary>
    public int DefaultSteps { get; }

    /// <summary>Denoises the latents of the audio (the autoencoder's decoder not applied).</summary>
    /// <param name="request">The request.</param>
    /// <param name="progress">Called after every step with the number of finished steps and the total.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>Generates the audio.</summary>
    /// <param name="request">The request.</param>
    /// <param name="progress">Called after every step with the number of finished steps and the total.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public Task<AudioBuffer> GenerateAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null,
        CancellationToken cancellationToken = default);
}
