using System.Runtime.CompilerServices;
using Avalonia.Media.Imaging;

namespace Sharplight.Studio.Converters;

/// <summary>
/// Bitmaps decoded from image bytes, kept as long as the bytes live: a worker decodes a generated image or preview before handing its
/// bytes to the UI (<see cref="Prepare"/>), so <see cref="BytesToBitmapConverter"/> finds it decoded instead of decoding on the UI thread,
/// and an image shown again (a re-templated list item) is not decoded twice.
/// </summary>
internal static class DecodedBitmaps
{
    private static readonly ConditionalWeakTable<byte[], Dictionary<int, Bitmap>> s_bitmaps = [];

    /// <summary>
    /// Decodes <paramref name="bytes"/> (any thread) so the converter finds the bitmap ready. Best effort: bytes that do not decode, or no
    /// Avalonia platform (view-model tests), leave the decoding to the converter.
    /// </summary>
    /// <param name="bytes">The encoded image.</param>
    /// <param name="width">The width to decode to, or 0 for the full size (the converter's parameter).</param>
    public static void Prepare(byte[] bytes, int width = 0)
    {
        try
        {
            Get(bytes, width);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // Decoded (or reported) by the converter when the image is shown.
        }
    }

    /// <summary>The bitmap of <paramref name="bytes"/> at <paramref name="width"/>, decoded on first use.</summary>
    /// <param name="bytes">The encoded image.</param>
    /// <param name="width">The width to decode to, or 0 for the full size.</param>
    /// <returns>The bitmap.</returns>
    /// <exception cref="ArgumentException">The bytes are not an image Avalonia decodes.</exception>
    public static Bitmap Get(byte[] bytes, int width)
    {
        var sizes = s_bitmaps.GetValue(bytes, _ => []);
        lock (sizes)
        {
            if (sizes.TryGetValue(width, out var cached))
            {
                return cached;
            }
        }

        using var stream = new MemoryStream(bytes);
        var bitmap = width > 0 ? Bitmap.DecodeToWidth(stream, width) : new Bitmap(stream);
        lock (sizes)
        {
            if (sizes.TryGetValue(width, out var raced))
            {
                bitmap.Dispose();
                return raced;
            }

            sizes[width] = bitmap;
            return bitmap;
        }
    }
}
