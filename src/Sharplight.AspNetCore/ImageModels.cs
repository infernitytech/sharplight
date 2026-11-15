using Sharplight.ImageGeneration;

namespace Sharplight.AspNetCore;

/// <summary>The image generators a host serves on <c>/v1/images/generations</c>, by the name clients pass as <c>model</c>.</summary>
public sealed class ImageModels
{
    /// <summary>Image generators by name.</summary>
    public Dictionary<string, IImageGenerator> Generators { get; } = new(StringComparer.Ordinal);

    /// <summary>Generator used when a request names none or an unknown one while only one is registered (default: the first registered).</summary>
    public string? DefaultGenerator { get; set; }

    /// <summary>Most images in one request (<c>n</c>).</summary>
    public int MaxImages { get; set; } = 4;

    /// <summary>Largest accepted request body.</summary>
    public long MaxRequestBodyBytes { get; set; } = 1024 * 1024;
}
