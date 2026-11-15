namespace Sharplight.GenerateImage;

/// <summary>The parsed command line of <c>sharplight-generate-image</c>.</summary>
internal sealed record GenerateImageSettings
{
    public required string Model { get; init; }

    public required string Prompt { get; init; }

    public string? NegativePrompt { get; init; }

    public required string Output { get; init; }

    public int? Width { get; init; }

    public int? Height { get; init; }

    public int? Steps { get; init; }

    public float? Guidance { get; init; }

    public long? Seed { get; init; }

    public int Count { get; init; } = 1;

    public string? Sampler { get; init; }

    public string? InitImage { get; init; }

    public string? Mask { get; init; }

    public float Strength { get; init; } = 0.75f;

    public string? Noise { get; init; }

    public int PreviewInterval { get; init; }

    public bool Cpu { get; init; }

    public int Device { get; init; }
}
