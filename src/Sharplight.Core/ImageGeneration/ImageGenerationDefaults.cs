namespace Sharplight.ImageGeneration;

/// <summary>
/// The values an image generator uses for the options a request leaves unset (<see cref="ImageGenerationOptions"/>): the model's native
/// size, its default number of steps, guidance scale and sampler. Distilled models (SDXL-Turbo, SDXL-Lightning, FLUX.1) differ from the usual
/// 25 steps with guidance 7.5, so a front end that shows these options reads them here; get it with
/// <c>generator.GetService(typeof(ImageGenerationDefaults))</c>.
/// </summary>
/// <param name="Width">Native image width in pixels.</param>
/// <param name="Height">Native image height in pixels.</param>
/// <param name="Steps">Default number of denoising steps.</param>
/// <param name="GuidanceScale">Default classifier-free guidance scale (1 or less: unguided).</param>
/// <param name="Sampler">Default sampler (<see cref="Samplers"/> names).</param>
/// <param name="ClassifierFreeGuidance">
/// Whether the guidance scale runs classifier-free guidance; false for guidance-distilled models (FLUX.1-dev: the scale is a denoiser
/// input, best left at the default) and models without guidance (FLUX.1-schnell).
/// </param>
public sealed record ImageGenerationDefaults(int Width, int Height, int Steps, float GuidanceScale, string Sampler, bool ClassifierFreeGuidance = true);
