namespace Sharplight.Architectures.SeedVr;

/// <summary>One attention window: the half-open token ranges [start, end) along frames, rows and columns of the patch grid.</summary>
/// <param name="FrameStart">First frame.</param>
/// <param name="FrameEnd">One past the last frame.</param>
/// <param name="RowStart">First row.</param>
/// <param name="RowEnd">One past the last row.</param>
/// <param name="ColumnStart">First column.</param>
/// <param name="ColumnEnd">One past the last column.</param>
internal sealed record VideoWindow(int FrameStart, int FrameEnd, int RowStart, int RowEnd, int ColumnStart, int ColumnEnd)
{
    /// <summary>Frames in the window.</summary>
    public int Frames => FrameEnd - FrameStart;

    /// <summary>Rows in the window.</summary>
    public int Rows => RowEnd - RowStart;

    /// <summary>Columns in the window.</summary>
    public int Columns => ColumnEnd - ColumnStart;

    /// <summary>Tokens in the window.</summary>
    public int Size => Frames * Rows * Columns;
}
