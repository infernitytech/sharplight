namespace Sharplight.Studio.Models;

/// <summary>What an image generation run was given and produced (<c>parameters.json</c> of a run, and the text of the PNGs' <c>parameters</c> entry).</summary>
internal sealed class ImageParametersFile
{
    /// <summary>The model that generated the images.</summary>
    public string Model { get; set; } = "";

    /// <summary>The prompt.</summary>
    public string Prompt { get; set; } = "";

    /// <summary>The negative prompt, or empty.</summary>
    public string NegativePrompt { get; set; } = "";

    /// <summary>The seed of the first image (image i used <c>Seed + i</c>).</summary>
    public long Seed { get; set; }

    /// <summary>Denoising steps.</summary>
    public int Steps { get; set; }

    /// <summary>Guidance scale.</summary>
    public float Guidance { get; set; }

    /// <summary>The sampler's name.</summary>
    public string Sampler { get; set; } = "";

    /// <summary>Image width in pixels.</summary>
    public int Width { get; set; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; set; }

    /// <summary>The strength of an image-to-image run (the input's share that is replaced), or null for text to image.</summary>
    public float? Strength { get; set; }

    /// <summary>Whether a mask limited the change (inpainting).</summary>
    public bool Inpainting { get; set; }

    /// <summary>The number of images.</summary>
    public int Count { get; set; }

    /// <summary>The page's parameter values, as a preset stores them (restore the page's panel with them).</summary>
    public Dictionary<string, string> Options { get; set; } = [];
}
