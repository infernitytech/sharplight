namespace Sharplight.GenerateVideo;

/// <summary>The parsed command line of <c>sharplight-generate-video</c>.</summary>
internal sealed record GenerateVideoSettings
{
    public required string Model { get; init; }

    public string? Companions { get; init; }

    public string? SecondTransformer { get; init; }

    public float? Guidance2 { get; init; }

    public required string Prompt { get; init; }

    public string? NegativePrompt { get; init; }

    public string? Image { get; init; }

    public required string Output { get; init; }

    public int? Width { get; init; }

    public int? Height { get; init; }

    public int? Frames { get; init; }

    public double? FramesPerSecond { get; init; }

    public int? Steps { get; init; }

    public float? Guidance { get; init; }

    public long? Seed { get; init; }

    public string? Sampler { get; init; }

    public string? Noise { get; init; }

    public bool Cpu { get; init; }

    public int Device { get; init; }

    public bool ReducedPrecision { get; init; }
}
