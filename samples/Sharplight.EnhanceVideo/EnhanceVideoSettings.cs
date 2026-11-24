using Sharplight.VideoEnhancement;

namespace Sharplight.EnhanceVideo;

/// <summary>The parsed command line of <c>sharplight-enhance-video</c>.</summary>
internal sealed record EnhanceVideoSettings
{
    public required string Model { get; init; }

    public required string Input { get; init; }

    public required string Output { get; init; }

    public int? Width { get; init; }

    public int? Height { get; init; }

    public float? Scale { get; init; }

    public int? FramesPerBatch { get; init; }

    public VideoColorCorrection? ColorCorrection { get; init; }

    public long? Seed { get; init; }

    public double Start { get; init; }

    public int? MaxFrames { get; init; }

    public bool Cpu { get; init; }

    public int Device { get; init; }

    public bool ReducedPrecision { get; init; }
}
