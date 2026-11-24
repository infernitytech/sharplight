namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// The window partitions of SeedVR2's attention (the official <c>720pwin_by_size_bysize</c> and <c>720pswin_by_size_bysize</c>): the window
/// size comes from splitting the token grid, rescaled to the area of the 720p reference grid, into the configured number of windows per
/// axis (frames capped at the reference frame count), and is then laid over the actual grid; the shifted layout moves the windows by half a
/// window along every axis that holds more than one window. Edge windows are smaller. Windows are listed column-major (columns outermost,
/// then rows, then frames), the order of the official implementation.
/// </summary>
internal static class SeedVrWindowLayout
{
    /// <summary>The windows of a <paramref name="frames"/> × <paramref name="rows"/> × <paramref name="columns"/> token grid.</summary>
    /// <param name="frames">Frames of the grid.</param>
    /// <param name="rows">Rows of the grid.</param>
    /// <param name="columns">Columns of the grid.</param>
    /// <param name="hp">The window counts, reference grid and frame cap.</param>
    /// <param name="shifted">Whether to lay out the shifted windows (odd layers).</param>
    public static IReadOnlyList<VideoWindow> Windows(int frames, int rows, int columns, SeedVrHyperParameters hp, bool shifted)
    {
        double scale = Math.Sqrt((double)hp.WindowReferenceGrid[0] * hp.WindowReferenceGrid[1] / ((double)rows * columns));
        // Python's round(): half to even.
        int resizedRows = (int)Math.Round(rows * scale, MidpointRounding.ToEven);
        int resizedColumns = (int)Math.Round(columns * scale, MidpointRounding.ToEven);
        int windowRows = CeilDiv(resizedRows, hp.WindowCounts[1]);
        int windowColumns = CeilDiv(resizedColumns, hp.WindowCounts[2]);
        int windowFrames = CeilDiv(Math.Min(frames, hp.WindowMaxFrames), hp.WindowCounts[0]);
        var frameRanges = shifted ? ShiftedRanges(frames, windowFrames) : Ranges(frames, windowFrames);
        var rowRanges = shifted ? ShiftedRanges(rows, windowRows) : Ranges(rows, windowRows);
        var columnRanges = shifted ? ShiftedRanges(columns, windowColumns) : Ranges(columns, windowColumns);
        var windows = new List<VideoWindow>(frameRanges.Count * rowRanges.Count * columnRanges.Count);
        foreach (var (c0, c1) in columnRanges)
        {
            foreach (var (r0, r1) in rowRanges)
            {
                foreach (var (f0, f1) in frameRanges)
                {
                    windows.Add(new VideoWindow(f0, f1, r0, r1, c0, c1));
                }
            }
        }

        return windows;
    }

    /// <summary>Consecutive non-empty ranges of <paramref name="size"/> tokens over an axis of <paramref name="length"/>.</summary>
    private static List<(int Start, int End)> Ranges(int length, int size)
    {
        var ranges = new List<(int, int)>();
        int count = CeilDiv(length, size);
        for (int i = 0; i < count; i++)
        {
            int start = i * size;
            int end = Math.Min((i + 1) * size, length);
            if (end > start)
            {
                ranges.Add((start, end));
            }
        }

        return ranges;
    }

    /// <summary>The ranges moved by half a window when the axis holds more than one window (clipped at both ends).</summary>
    private static List<(int Start, int End)> ShiftedRanges(int length, int size)
    {
        double shift = size < length ? 0.5 : 0;
        int count = shift > 0 ? (int)Math.Ceiling((length - shift) / size) + 1 : 1;
        var ranges = new List<(int, int)>();
        for (int i = 0; i < count; i++)
        {
            // Python's int() truncates toward zero.
            int start = Math.Max((int)Math.Truncate((i - shift) * size), 0);
            int end = Math.Min((int)Math.Truncate((i - shift + 1) * size), length);
            if (end > start)
            {
                ranges.Add((start, end));
            }
        }

        return ranges;
    }

    private static int CeilDiv(int value, int divisor) => (value + divisor - 1) / divisor;
}
