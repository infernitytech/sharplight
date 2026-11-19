using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Stable Audio 3 model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.StableAudio</c>).</summary>
/// <param name="Objective">The diffusion objective: <c>rectified_flow</c> (the base models, sampled with Euler steps and guidance) or <c>rf_denoiser</c> (the
/// post-trained models, sampled with ping-pong steps).</param>
/// <param name="Hidden">Width of the diffusion transformer.</param>
/// <param name="Layers">Its layers.</param>
/// <param name="Heads">Its attention heads.</param>
/// <param name="Differential">Whether its attention is differential (two softmax maps, the second subtracted).</param>
/// <param name="MemoryTokens">The learned rows that join the latent frames in self-attention.</param>
/// <param name="RopeDims">The rotated head dimensions (partial rotary embedding).</param>
/// <param name="TimeFeatures">Width of the exponential Fourier features of the time and the seconds.</param>
/// <param name="TimeLogSnr">Whether the network's time is the normalized log-SNR of t rather than t.</param>
/// <param name="ConditionWidth">Width of the conditioning rows (the prompt rows and the seconds row).</param>
/// <param name="PromptLength">The fixed prompt length in tokens.</param>
/// <param name="LocalChannels">Channels of the local additive conditioning (the inpainting mask and the masked latents).</param>
/// <param name="MinSeconds">The smallest number of seconds the seconds conditioner encodes.</param>
/// <param name="MaxSeconds">The largest.</param>
/// <param name="SampleRate">Sample rate of the decoded audio.</param>
/// <param name="AudioChannels">Channels of the decoded audio.</param>
/// <param name="LatentChannels">Values of a latent frame.</param>
/// <param name="SamplesPerFrame">Audio samples per latent frame.</param>
/// <param name="MaxFrames">The longest latent sequence (the model's sample size).</param>
/// <param name="FrameAlign">The latent frames are a multiple of this (the autoencoder's chunks).</param>
/// <param name="HeadroomSeconds">Seconds of latents generated after the requested duration.</param>
/// <param name="DefaultSteps">Default number of sampler steps.</param>
/// <param name="DefaultGuidance">Default classifier-free guidance scale.</param>
internal sealed record StableAudio3Parameters(string Objective, int Hidden, int Layers, int Heads, bool Differential, int MemoryTokens, int RopeDims, int TimeFeatures,
    bool TimeLogSnr, int ConditionWidth, int PromptLength, int LocalChannels, float MinSeconds, float MaxSeconds, int SampleRate, int AudioChannels, int LatentChannels,
    int SamplesPerFrame, int MaxFrames, int FrameAlign, float HeadroomSeconds, int DefaultSteps, float DefaultGuidance)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "stable_audio_3";

    /// <summary>The key prefix of the hyper-parameters.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>Norm epsilon of the transformer's RMS norms.</summary>
    public const float NormEps = 1e-5f;

    /// <summary>Epsilon of the query and key RMS norms.</summary>
    public const float QkNormEps = 1e-6f;

    /// <summary>Rotary base.</summary>
    public const float RopeBase = 10000f;

    /// <summary>The lowest frequency of the exponential Fourier features.</summary>
    public const float MinFrequency = 0.5f;

    /// <summary>The highest.</summary>
    public const float MaxFrequency = 10000f;

    /// <summary>Width of a head.</summary>
    public int HeadDim => Hidden / Heads;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static StableAudio3Parameters From(ModelSource source) => new(
        source.Arch("objective", "rectified_flow"), source.Arch<int>("hidden_size"), source.Arch<int>("layer_count"), source.Arch<int>("head_count"),
        source.Arch("differential", false), source.Arch("memory_tokens", 0), source.Arch<int>("rope_dims"), source.Arch("time_features", 256), source.Arch("time_logsnr", false),
        source.Arch<int>("condition_width"), source.Arch<int>("prompt_length"), source.Arch("local_channels", 0), source.Arch("seconds.min", 0f), source.Arch<float>("seconds.max"),
        source.Arch<int>("sample_rate"), source.Arch<int>("audio_channels"), source.Arch<int>("latent_channels"), source.Arch<int>("samples_per_frame"),
        source.Arch<int>("max_frames"), source.Arch("frame_align", 1), source.Arch("headroom_seconds", 6f), source.Arch("default_steps", 50), source.Arch("default_guidance", 7f));
}
