namespace Sharplight.Vision;

/// <summary>The kinds of prompt point of the promptable segmentation contract (<see cref="Models.WellKnown.SegmentDecodeGraph"/>, input <c>point_kinds</c>).</summary>
public static class SegmentPromptKinds
{
    /// <summary>A point on the background (not part of the object).</summary>
    public const int Negative = 0;

    /// <summary>A point on the object.</summary>
    public const int Positive = 1;

    /// <summary>The top-left corner of a box.</summary>
    public const int BoxStart = 2;

    /// <summary>The bottom-right corner of a box.</summary>
    public const int BoxEnd = 3;

    /// <summary>A padding entry without coordinates (completes a prompt that has no box).</summary>
    public const int None = 4;
}
