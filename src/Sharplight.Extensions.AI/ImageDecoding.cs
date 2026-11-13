using Microsoft.Extensions.AI;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

internal static class ImageDecoding
{
    public static ImageInput DecodeWithStb(DataContent content)
    {
        var img = StbImageSharp.ImageResult.FromMemory(content.Data.ToArray(), StbImageSharp.ColorComponents.RedGreenBlue);
        return new ImageInput(img.Width, img.Height, img.Data);
    }
}
