using Sharplight.Samples;
using Sharplight.Vision;
using StbImageSharp;

namespace Sharplight.Cli;

/// <summary>Decodes image files into the raw RGB pixels the vision encoder takes.</summary>
internal static class ImageLoader
{
    public static List<ImageInput> Load(IEnumerable<string> paths)
    {
        var images = new List<ImageInput>();
        foreach (var path in paths)
        {
            images.Add(LoadOne(path));
        }

        return images;
    }

    private static ImageInput LoadOne(string path)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Image '{path}' does not exist.");
        }

        try
        {
            var image = ImageResult.FromMemory(File.ReadAllBytes(path), ColorComponents.RedGreenBlue);
            return new ImageInput(image.Width, image.Height, image.Data);
        }
        catch (InvalidOperationException error)
        {
            throw new SampleException($"Cannot decode image '{path}': {error.Message}", error);
        }
    }
}
