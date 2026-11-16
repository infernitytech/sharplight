namespace Sharplight.Studio.Services.Gallery;

/// <summary>A LoRA that was merged into the model of a stored image.</summary>
internal sealed record GalleryLora
{
    /// <summary>The adapter's name (file name without extension).</summary>
    public string Name { get; init; } = "";

    /// <summary>The strength it was merged with.</summary>
    public float Scale { get; init; } = 1;

    /// <summary>The LoRA file, to attach it again when reusing the settings.</summary>
    public string Path { get; init; } = "";
}
