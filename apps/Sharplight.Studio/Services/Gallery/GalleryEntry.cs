namespace Sharplight.Studio.Services.Gallery;

/// <summary>One stored generation: the JSON sidecar next to its PNG (the image, or the first frame of a video) in the gallery.</summary>
internal sealed record GalleryEntry
{
    /// <summary>A unique id, also the file name (without extension) of the PNG and its sidecar.</summary>
    public string Id { get; init; } = "";

    /// <summary>When the image was generated.</summary>
    public DateTimeOffset Created { get; init; } = DateTimeOffset.Now;

    /// <summary>The model id.</summary>
    public string Model { get; init; } = "";

    /// <summary>The prompt.</summary>
    public string Prompt { get; init; } = "";

    /// <summary>The negative prompt, or empty.</summary>
    public string NegativePrompt { get; init; } = "";

    /// <summary>The seed of this image.</summary>
    public long Seed { get; init; }

    /// <summary>Denoising steps.</summary>
    public int Steps { get; init; }

    /// <summary>Guidance scale.</summary>
    public float Guidance { get; init; }

    /// <summary>Width in pixels.</summary>
    public int Width { get; init; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; init; }

    /// <summary>The sampler, or empty for the model's own.</summary>
    public string Sampler { get; init; } = "";

    /// <summary>The LoRA adapters merged into the model, if any.</summary>
    public List<GalleryLora> Loras { get; init; } = [];

    /// <summary>Seconds the whole run took (all images of the run).</summary>
    public double Seconds { get; init; }

    /// <summary>The parameter panel's values, to refill the form.</summary>
    public Dictionary<string, string> Options { get; init; } = [];

    /// <summary>Frames of a stored video (0 for an image); the PNG is then its first frame.</summary>
    public int Frames { get; init; }

    /// <summary>Frames per second of a stored video.</summary>
    public double FramesPerSecond { get; init; }

    /// <summary>The video file next to the PNG (<c>&lt;id&gt;.webm</c> or <c>.gif</c>), or empty when none could be encoded.</summary>
    public string Video { get; init; } = "";

    /// <summary>Path of the PNG relative to the gallery folder (<c>2026-11-16/&lt;id&gt;.png</c>); filled by the store.</summary>
    public string File { get; init; } = "";
}
