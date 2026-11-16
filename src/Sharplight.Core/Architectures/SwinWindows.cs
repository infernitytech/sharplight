namespace Sharplight.Architectures;

/// <summary>
/// The constants of Swin window attention over a height × width grid of rows: window-major gather indices (with zero padding to a multiple of the window,
/// a cyclic shift of the padded grid), the relative position index and the additive mask of shifted windows.
/// </summary>
internal static class SwinWindows
{
    /// <summary>The grid side rounded up to a multiple of <paramref name="window"/>.</summary>
    public static int Padded(int length, int window) => (length + window - 1) / window * window;

    /// <summary>
    /// For a height × width grid padded to whole windows of side <paramref name="window"/> and cyclically shifted by -<paramref name="shift"/>: the source row of every
    /// window-major slot (<c>height · width</c> addresses the zero row that stands for padding) and the slot of every unpadded row once the shift is undone.
    /// </summary>
    public static (int[] Into, int[] Back) Indices(int height, int width, int window, int shift)
    {
        int hp = Padded(height, window);
        int wp = Padded(width, window);
        var into = new int[hp * wp];
        var back = new int[height * width];
        int perRow = wp / window;
        for (int r = 0; r < hp; r++)
        {
            for (int c = 0; c < wp; c++)
            {
                int slot = (((r / window * perRow) + (c / window)) * window * window) + (r % window * window) + (c % window);
                int sr = (r + shift) % hp;
                int sc = (c + shift) % wp;
                into[slot] = sr < height && sc < width ? (sr * width) + sc : height * width;
                if (sr < height && sc < width)
                {
                    back[(sr * width) + sc] = slot;
                }
            }
        }

        return (into, back);
    }

    /// <summary>Index into the (2w-1)² relative position table of every (query, key) pair of a window.</summary>
    public static int[] RelativeIndex(int window)
    {
        int n = window * window;
        var index = new int[n * n];
        for (int q = 0; q < n; q++)
        {
            for (int key = 0; key < n; key++)
            {
                int dy = (q / window) - (key / window) + window - 1;
                int dx = (q % window) - (key % window) + window - 1;
                index[(q * n) + key] = (dy * ((2 * window) - 1)) + dx;
            }
        }

        return index;
    }

    /// <summary>The additive mask [windows, 1, N, N] of shifted windows over the padded grid: -100 between positions that came from different regions of the unshifted grid.</summary>
    public static float[] ShiftMask(int paddedHeight, int paddedWidth, int window, int shift)
    {
        var region = new int[paddedHeight * paddedWidth];
        int count = 0;
        (int Start, int End)[] spans(int length) => [(0, length - window), (length - window, length - shift), (length - shift, length)];
        foreach (var (y0, y1) in spans(paddedHeight))
        {
            foreach (var (x0, x1) in spans(paddedWidth))
            {
                for (int y = y0; y < y1; y++)
                {
                    for (int x = x0; x < x1; x++)
                    {
                        region[(y * paddedWidth) + x] = count;
                    }
                }

                count++;
            }
        }

        int n = window * window;
        int perRow = paddedWidth / window;
        int windows = paddedHeight / window * perRow;
        var mask = new float[windows * n * n];
        for (int win = 0; win < windows; win++)
        {
            for (int q = 0; q < n; q++)
            {
                int qr = region[(((win / perRow * window) + (q / window)) * paddedWidth) + ((win % perRow) * window) + (q % window)];
                for (int key = 0; key < n; key++)
                {
                    int kr = region[(((win / perRow * window) + (key / window)) * paddedWidth) + ((win % perRow) * window) + (key % window)];
                    mask[(((win * n) + q) * n) + key] = qr == kr ? 0f : -100f;
                }
            }
        }

        return mask;
    }
}
