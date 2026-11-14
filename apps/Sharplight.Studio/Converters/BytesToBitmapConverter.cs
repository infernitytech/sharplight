using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;

namespace Sharplight.Studio.Converters;

/// <summary>Decodes image bytes into a bitmap (thumbnail width when a parameter gives one); null for anything else.</summary>
internal sealed class BytesToBitmapConverter : IValueConverter
{
    /// <summary>The shared instance.</summary>
    public static BytesToBitmapConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not byte[] { Length: > 0 } bytes)
        {
            return null;
        }

        try
        {
            using var stream = new MemoryStream(bytes);
            return parameter is string text && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int width)
                ? Bitmap.DecodeToWidth(stream, width)
                : new Bitmap(stream);
        }
        catch (Exception error) when (error is ArgumentException or InvalidOperationException or IOException or NotSupportedException)
        {
            return null;
        }
    }

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}
