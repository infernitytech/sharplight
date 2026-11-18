namespace Sharplight.Vision;

/// <summary>The resampling filter of an image resize (the names of Pillow's <c>Image.Resampling</c>; shrinking is antialiased).</summary>
public enum ImageResampleFilter
{
    /// <summary>Nearest neighbour.</summary>
    Nearest,

    /// <summary>Bilinear (triangle kernel, widened when shrinking).</summary>
    Bilinear,

    /// <summary>Bicubic (Keys kernel with a = -0.5, widened when shrinking).</summary>
    Bicubic,

    /// <summary>Lanczos (windowed sinc with three lobes, widened when shrinking; Pillow's <c>LANCZOS</c>, diffusers' default image resize).</summary>
    Lanczos,
}
