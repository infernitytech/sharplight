namespace Sharplight.Vision;

/// <summary>What to segment: points on (or beside) the object and/or a box around it, in pixels of the image.</summary>
public sealed record SegmentationPrompt
{
    /// <summary>The point prompts.</summary>
    public IReadOnlyList<SegmentationPoint> Points { get; init; } = [];

    /// <summary>A box around the object, or <see langword="null"/>.</summary>
    public BoundingBox? Box { get; init; }

    /// <summary>A prompt of one positive point.</summary>
    /// <param name="x">Horizontal position.</param>
    /// <param name="y">Vertical position.</param>
    public static SegmentationPrompt FromPoint(float x, float y) => new() { Points = [new SegmentationPoint(x, y)] };

    /// <summary>A prompt of a box.</summary>
    /// <param name="box">The box.</param>
    public static SegmentationPrompt FromBox(BoundingBox box) => new() { Box = box };
}
