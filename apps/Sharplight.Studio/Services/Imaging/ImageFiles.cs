using System.Diagnostics.CodeAnalysis;
using Avalonia.Media.Imaging;
using Sharplight.Imaging;
using Sharplight.Studio.Services.Platform;
using Sharplight.Vision;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>Reads and writes the image files of the vision and image pages: PNG through the engine's codec, other formats through Avalonia's decoders.</summary>
internal static class ImageFiles
{
    /// <summary>The file extensions the pages accept.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"];

    /// <summary>The filter of the file picker.</summary>
    public static FileFilter Filter { get; } = new("Images", [.. Extensions.Select(extension => "*" + extension)]);

    /// <summary>Whether the path names a file type the pages read.</summary>
    /// <param name="path">The file.</param>
    public static bool IsImage(string path) => Extensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    /// <summary>Reads an image file as 8-bit RGB (transparency is dropped).</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not an image this machine can decode.</exception>
    public static ImageInput Read(string path)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(path);
        }
        catch (UnauthorizedAccessException error)
        {
            throw new IOException(error.Message, error);
        }

        return Decode(bytes);
    }

    /// <summary>Decodes the bytes of an image file as 8-bit RGB.</summary>
    /// <param name="bytes">The file's contents.</param>
    /// <exception cref="InvalidDataException">The data is not an image this machine can decode.</exception>
    public static ImageInput Decode(byte[] bytes)
    {
        try
        {
            return PngCodec.Decode(bytes);
        }
        catch (Exception error) when (error is InvalidDataException or NotSupportedException)
        {
            // Not a PNG the engine's decoder reads (a JPEG, or a PNG with 16 bits or interlacing): the platform decoders may.
            return DecodeWithPlatform(bytes, error);
        }
    }

    /// <summary>Encodes an image as PNG.</summary>
    /// <param name="image">The image.</param>
    public static byte[] ToPng(ImageInput image) => PngCodec.Encode(image, System.IO.Compression.CompressionLevel.Fastest);

    [SuppressMessage("Design", "CA1031", Justification = "Platform decoders throw several exception types for damaged files.")]
    private static ImageInput DecodeWithPlatform(byte[] bytes, Exception first)
    {
        try
        {
            using var input = new MemoryStream(bytes);
            using var bitmap = new Bitmap(input);
            using var output = new MemoryStream();
            bitmap.Save(output);
            return PngCodec.Decode(output.ToArray());
        }
        catch (Exception)
        {
            throw new InvalidDataException("The file is not an image this machine can decode (PNG, JPEG, WebP, BMP and GIF are supported).", first);
        }
    }
}
