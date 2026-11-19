using Sharplight.Llm.Serving.Chat;
using Sharplight.Vision;

namespace Sharplight.Hosting.Loaders;

/// <summary>Decodes the image parts of chat requests (PNG, JPEG, BMP, GIF, TGA, PSD) to RGB pixels for models served with image input.</summary>
internal static class ChatImageDecoder
{
    /// <summary>The pixels of an inline image.</summary>
    /// <exception cref="ArgumentException">The image has no inline bytes.</exception>
    public static ImageInput Decode(ImageData image)
    {
        var bytes = image.Bytes ?? throw new ArgumentException("The image has no inline data.", nameof(image));
        var decoded = StbImageSharp.ImageResult.FromMemory(bytes, StbImageSharp.ColorComponents.RedGreenBlue);
        return new ImageInput(decoded.Width, decoded.Height, decoded.Data);
    }
}
