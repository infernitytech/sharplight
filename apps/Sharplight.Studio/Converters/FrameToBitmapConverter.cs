using System.Globalization;
using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Sharplight.Vision;

namespace Sharplight.Studio.Converters;

/// <summary>Turns a decoded video frame (8-bit RGB) into a bitmap for the player; null for anything else.</summary>
internal sealed class FrameToBitmapConverter : IValueConverter
{
    private const int BytesPerPixel = 4;

    /// <summary>The shared instance.</summary>
    public static FrameToBitmapConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        => value is ImageInput { Width: > 0, Height: > 0 } frame ? ToBitmap(frame) : null;

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();

    private static WriteableBitmap ToBitmap(ImageInput frame)
    {
        var bitmap = new WriteableBitmap(new PixelSize(frame.Width, frame.Height), new Vector(96, 96), PixelFormat.Bgra8888, AlphaFormat.Opaque);
        using var buffer = bitmap.Lock();
        var row = new byte[frame.Width * BytesPerPixel];
        for (int y = 0; y < frame.Height; y++)
        {
            int source = y * frame.Width * 3;
            for (int x = 0; x < frame.Width; x++)
            {
                int target = x * BytesPerPixel;
                row[target] = frame.Rgb[source + 2];
                row[target + 1] = frame.Rgb[source + 1];
                row[target + 2] = frame.Rgb[source];
                row[target + 3] = 255;
                source += 3;
            }

            Marshal.Copy(row, 0, buffer.Address + y * buffer.RowBytes, row.Length);
        }

        return bitmap;
    }
}
