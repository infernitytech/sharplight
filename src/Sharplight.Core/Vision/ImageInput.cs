namespace Sharplight.Vision;

/// <summary>A decoded image: 8-bit interleaved RGB (row-major, 3 bytes per pixel). File decoding is the host's job.</summary>
public sealed record ImageInput(int Width, int Height, byte[] Rgb)
{
    internal void Validate()
    {
        if (Width <= 0 || Height <= 0)
        {
            throw new ArgumentException("Image has no pixels.");
        }

        if (Rgb.LongLength != 3L * Width * Height)
        {
            throw new ArgumentException($"Expected {3L * Width * Height} RGB bytes for {Width}x{Height}, got {Rgb.Length}.");
        }
    }
}
