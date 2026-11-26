namespace Sharplight.ImageGeneration;

/// <summary>Per-request settings of <see cref="IImageGenerator"/>; <see langword="null"/> values take the model's defaults.</summary>
public sealed record ImageGenerationOptions
{
    /// <summary>Output width in pixels, or <see langword="null"/> for the model's native size.</summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels, or <see langword="null"/> for the model's native size.</summary>
    public int? Height { get; init; }

    /// <summary>Number of denoising steps, or <see langword="null"/> for the model's default.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale, or <see langword="null"/> for the model's default (distilled models ignore it).</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>
    /// For InstructPix2Pix-style models (<see cref="ImageGenerationDefaults.ImageGuidanceScale"/> above 0): how strongly the output follows
    /// <see cref="ImageGenerationRequest.Image"/> (1: not at all, higher: closer), or <see langword="null"/> for the model's default.
    /// </summary>
    public float? ImageGuidanceScale { get; init; }

    /// <summary>Seed of the initial noise; <see langword="null"/> picks one (reported in <see cref="ImageGenerationResult.Seed"/>).</summary>
    public long? Seed { get; init; }

    /// <summary>Number of images to generate.</summary>
    public int Count { get; init; } = 1;

    /// <summary>Scheduler (sampler) name understood by the implementation, or <see langword="null"/> for the model's configured one.</summary>
    public string? Scheduler { get; init; }

    /// <summary>
    /// Explicit initial noise instead of the seeded noise: unit Gaussian values of all images, image after image, each in the model's
    /// latent layout (for the latent-diffusion pipeline [channels, height / 8, width / 8]); used to reproduce another implementation's
    /// run exactly. <see langword="null"/> draws the noise from <see cref="Seed"/>.
    /// </summary>
    public float[]? InitialNoise { get; init; }

    /// <summary>
    /// Every how many steps the streaming call attaches a low-resolution preview of the current estimate to its update
    /// (<see cref="ImageGenerationUpdate.Preview"/>); 0 for none. Previews are cheap: decoded by the model's tiny autoencoder
    /// (TAESD) at full size when it has one, otherwise the latent colours at latent resolution (no VAE decode; needs a model that describes them).
    /// </summary>
    public int PreviewInterval { get; init; }

    /// <summary>
    /// For models conditioned on an aesthetic score (the SDXL refiner, <see cref="ImageGenerationDefaults.AestheticScore"/> above 0): the score
    /// the prompt's row asks for, or <see langword="null"/> for the model's default (6).
    /// </summary>
    public float? AestheticScore { get; init; }

    /// <summary>
    /// For models conditioned on an aesthetic score: the score of the negative prompt's row, or <see langword="null"/> for the model's
    /// default (2.5).
    /// </summary>
    public float? NegativeAestheticScore { get; init; }

    /// <summary>
    /// A second generator that finishes the denoising (the SDXL refiner after the SDXL base, as diffusers' <c>denoising_end</c> and
    /// <c>denoising_start</c>): this generator runs the schedule until <see cref="RefinerStart"/>, hands its latents to the refiner, which
    /// runs the remaining steps of the same schedule on its own conditioning and decodes the image; or <see langword="null"/> for none. Both
    /// must share the latent space (the same VAE latents); the refiner is used with the request's prompts and this request's size, steps,
    /// guidance and sampler, and must not be this generator.
    /// </summary>
    public IImageGenerator? Refiner { get; init; }

    /// <summary>
    /// With <see cref="Refiner"/>: the fraction of the noise schedule (of the training timesteps, as diffusers' <c>denoising_end</c>) after
    /// which the refiner takes over, in (0, 1), or <see langword="null"/> for 0.8.
    /// </summary>
    public float? RefinerStart { get; init; }

    /// <summary>The default <see cref="RefinerStart"/>.</summary>
    public const float DefaultRefinerStart = 0.8f;

    /// <summary>
    /// Whether to decode the image (and encode an input image) with the model's tiny autoencoder (TAESD, loaded as its <c>tiny-vae</c>
    /// auxiliary source; <see cref="ImageGenerationDefaults.TinyAutoencoder"/>) instead of its VAE: a fraction of the VAE's time and memory,
    /// at a visibly lower fidelity (fine detail and text). Previews (<see cref="PreviewInterval"/>) use the tiny autoencoder whenever the
    /// model has one, regardless of this option.
    /// </summary>
    public bool TinyAutoencoder { get; init; }

    /// <summary>
    /// For models with a ControlNet (<see cref="ImageGenerationDefaults.ControlNet"/>): how strongly the control image steers the denoiser
    /// (the scale of the ControlNet's residuals, diffusers' <c>controlnet_conditioning_scale</c>), or <see langword="null"/> for 1.
    /// </summary>
    public float? ControlScale { get; init; }

    /// <summary>
    /// For models with a ControlNet: the fraction of the steps after which the control image starts to apply, in [0, 1) (diffusers'
    /// <c>control_guidance_start</c>), or <see langword="null"/> for 0.
    /// </summary>
    public float? ControlStart { get; init; }

    /// <summary>
    /// For models with a ControlNet: the fraction of the steps after which the control image stops applying, in (0, 1] (diffusers'
    /// <c>control_guidance_end</c>), or <see langword="null"/> for 1.
    /// </summary>
    public float? ControlEnd { get; init; }

    /// <summary>The defaults.</summary>
    public static ImageGenerationOptions Default { get; } = new();
}
