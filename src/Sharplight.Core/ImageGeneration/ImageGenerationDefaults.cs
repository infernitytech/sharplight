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
/// <param name="ReferenceImages">
/// How many reference images the model edits (<see cref="ImageGenerationRequest.ReferenceImages"/>; FLUX.1 Kontext 1, FLUX.2 up to 10), or 0
/// for models that do not edit.
/// </param>
/// <param name="ImageGuidanceScale">
/// The default <see cref="ImageGenerationOptions.ImageGuidanceScale"/> of InstructPix2Pix-style models, which edit the request's
/// <see cref="ImageGenerationRequest.Image"/> by an instruction; 0 for models without it.
/// </param>
/// <param name="AestheticScore">
/// The default <see cref="ImageGenerationOptions.AestheticScore"/> of models conditioned on an aesthetic score (the SDXL refiner, typically
/// the second stage of <see cref="ImageGenerationOptions.Refiner"/>); 0 for models without it.
/// </param>
public sealed record ImageGenerationDefaults(int Width, int Height, int Steps, float GuidanceScale, string Sampler, bool ClassifierFreeGuidance = true,
    int ReferenceImages = 0, float ImageGuidanceScale = 0f, float AestheticScore = 0f);
