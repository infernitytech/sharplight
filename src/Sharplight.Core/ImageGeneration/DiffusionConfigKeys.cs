namespace Sharplight.ImageGeneration;

/// <summary>
/// Configuration keys of image-generation model descriptions (<see cref="Models.ModelComponent.Config"/> and
/// <see cref="Models.ModelDescription.Config"/>) that the built-in latent-diffusion pipeline reads. Adapters write them; values the
/// adapter leaves out take the defaults named here.
/// </summary>
public static class DiffusionConfigKeys
{
    /// <summary>Model: native image width in pixels.</summary>
    public const string Width = "width";

    /// <summary>Model: native image height in pixels.</summary>
    public const string Height = "height";

    /// <summary>Model: default number of denoising steps (default 25).</summary>
    public const string Steps = "steps";

    /// <summary>Model: default classifier-free guidance scale (default 7.5; distilled models set 1).</summary>
    public const string GuidanceScale = "guidanceScale";

    /// <summary>Denoiser and VAE: latent channels.</summary>
    public const string LatentChannels = "latentChannels";

    /// <summary>VAE: pixels per latent along each axis (8 for the SD VAEs).</summary>
    public const string Downscale = "downscale";

    /// <summary>VAE: optional [latent channels × 3] row-major map from model latents to RGB in [-1, 1], for step previews.</summary>
    public const string LatentRgbFactors = "latentRgbFactors";

    /// <summary>Text encoder and tokenizer: tokens per prompt, including the start and end tokens (77 for CLIP).</summary>
    public const string ContextLength = "contextLength";

    /// <summary>Scheduler: number of training timesteps (default 1000).</summary>
    public const string TrainSteps = "trainSteps";

    /// <summary>Scheduler: first beta of the training schedule.</summary>
    public const string BetaStart = "betaStart";

    /// <summary>Scheduler: last beta of the training schedule.</summary>
    public const string BetaEnd = "betaEnd";

    /// <summary>Scheduler: <c>linear</c>, <c>scaled_linear</c> or <c>squaredcos_cap_v2</c>.</summary>
    public const string BetaSchedule = "betaSchedule";

    /// <summary>Scheduler: what the denoiser predicts, <c>epsilon</c>, <c>v_prediction</c> or <c>sample</c>.</summary>
    public const string PredictionType = "predictionType";

    /// <summary>Scheduler: spacing of the inference timesteps, <c>leading</c>, <c>trailing</c> or <c>linspace</c>.</summary>
    public const string TimestepSpacing = "timestepSpacing";

    /// <summary>Scheduler: offset added to <c>leading</c> timesteps.</summary>
    public const string StepsOffset = "stepsOffset";

    /// <summary>Scheduler: whether DDIM's last step goes to alpha 1 instead of the first training alpha.</summary>
    public const string SetAlphaToOne = "setAlphaToOne";

    /// <summary>Scheduler: whether the betas are rescaled to zero terminal SNR.</summary>
    public const string ZeroTerminalSnr = "zeroTerminalSnr";

    /// <summary>Scheduler: whether DDIM clips its clean-sample estimate to [-1, 1] (absent: the sampler's default, on).</summary>
    public const string ClipSample = "clipSample";

    /// <summary>Scheduler: the default sampler (<see cref="Samplers"/> names).</summary>
    public const string Sampler = "sampler";
}
