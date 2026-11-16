namespace Sharplight.VideoGeneration;

/// <summary>
/// The values a video generator uses for the options a request leaves unset (<see cref="VideoGenerationOptions"/>) and what it accepts;
/// a front end that shows these options reads them with <c>generator.GetService(typeof(VideoGenerationDefaults))</c>.
/// </summary>
/// <param name="Width">Default width in pixels.</param>
/// <param name="Height">Default height in pixels.</param>
/// <param name="Frames">Default frame count.</param>
/// <param name="FramesPerSecond">Native playback rate.</param>
/// <param name="Steps">Default number of denoising steps.</param>
/// <param name="GuidanceScale">Default classifier-free guidance scale.</param>
/// <param name="Sampler">Default sampler (<see cref="ImageGeneration.Samplers"/> names).</param>
/// <param name="SizeMultiple">Width and height are multiples of this many pixels.</param>
/// <param name="FrameStep">Frame counts are 1 + k · this (the VAE's temporal compression).</param>
/// <param name="SupportsImage">Whether the model generates from a start image (<see cref="VideoGenerationRequest.Image"/>).</param>
/// <param name="NegativePrompt">The negative prompt used when a request gives none (the model's published default), or <see langword="null"/>.</param>
/// <param name="RequiresImage">Whether the model generates only from a start image (no text-to-video).</param>
/// <param name="SupportsLastImage">Whether the model takes a last frame (<see cref="VideoGenerationRequest.LastImage"/>).</param>
public sealed record VideoGenerationDefaults(int Width, int Height, int Frames, double FramesPerSecond, int Steps, float GuidanceScale, string Sampler,
    int SizeMultiple, int FrameStep, bool SupportsImage, string? NegativePrompt, bool RequiresImage = false, bool SupportsLastImage = false);
