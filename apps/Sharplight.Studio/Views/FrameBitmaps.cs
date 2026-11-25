using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Sharplight.Vision;

namespace Sharplight.Studio.Views;

/// <summary>Copies 8-bit RGB frames into BGRA <see cref="WriteableBitmap"/>s for the video views (on any thread).</summary>
internal static class FrameBitmaps
{
    private const int BytesPerPixel = 4;

    /// <summary>Writes <paramref name="frame"/> into <paramref name="reuse"/> when it has the frame's size, else into a new bitmap.</summary>
    /// <param name="frame">The frame.</param>
    /// <param name="reuse">A bitmap of an earlier frame, or null.</param>
    /// <returns>The bitmap holding the frame.</returns>
    public static WriteableBitmap Fill(ImageInput frame, WriteableBitmap? reuse)
    {
        var size = new PixelSize(frame.Width, frame.Height);
        var bitmap = reuse is not null && reuse.PixelSize == size
            ? reuse
            : new WriteableBitmap(size, new Vector(96, 96), PixelFormat.Bgra8888, AlphaFormat.Opaque);
        using var buffer = bitmap.Lock();
        var row = new byte[frame.Width * BytesPerPixel];
        byte[] rgb = frame.Rgb;
        for (int y = 0; y < frame.Height; y++)
        {
            int source = y * frame.Width * 3;
            for (int x = 0; x < frame.Width; x++)
            {
                int target = x * BytesPerPixel;
                row[target] = rgb[source + 2];
                row[target + 1] = rgb[source + 1];
                row[target + 2] = rgb[source];
                row[target + 3] = 255;
                source += 3;
            }

            Marshal.Copy(row, 0, buffer.Address + y * buffer.RowBytes, row.Length);
        }

        return bitmap;
    }
}
