using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>What to generate (<see cref="IVideoGenerator"/>).</summary>
public sealed record VideoGenerationRequest
{
    /// <summary>The text prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>What the video should not show (classifier-free guidance's negative conditioning), or <see langword="null"/> for the model's default.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>
    /// The first frame for image-to-video generation (resized to the output size), or <see langword="null"/> for
    /// text-to-video. Needs a model that supports it (<see cref="VideoGenerationDefaults.SupportsImage"/>).
    /// </summary>
    public ImageInput? Image { get; init; }
}
