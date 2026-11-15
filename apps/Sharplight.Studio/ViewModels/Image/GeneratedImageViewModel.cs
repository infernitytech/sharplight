using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>One image of an image generation result: its PNG (with the prompt and parameters as text metadata), its pixels and its seed.</summary>
internal sealed class GeneratedImageViewModel
{
    /// <param name="number">The position in the result, from 1.</param>
    /// <param name="png">The PNG file's bytes, metadata included.</param>
    /// <param name="image">The decoded pixels.</param>
    /// <param name="seed">The seed of this image's noise.</param>
    public GeneratedImageViewModel(int number, byte[] png, ImageInput image, long seed)
    {
        Number = number;
        Png = png;
        Image = image;
        Seed = seed;
    }

    /// <summary>The position in the result, from 1.</summary>
    public int Number { get; }

    /// <summary>The PNG file's bytes, metadata included.</summary>
    public byte[] Png { get; }

    /// <summary>The decoded pixels.</summary>
    public ImageInput Image { get; }

    /// <summary>The seed of this image's noise.</summary>
    public long Seed { get; }

    /// <summary>The name of the image's file (<c>image-1.png</c>).</summary>
    public string FileName => $"image-{Number}.png";

    /// <summary>A short description for the thumbnail's tooltip.</summary>
    public string Caption => $"Image {Number} · seed {Seed} · {Image.Width} × {Image.Height}";
}
