namespace Sharplight.Imaging;

/// <summary>An image after <see cref="ImageProcessor.Process"/>: normalized planar floats plus the geometry that maps model coordinates back to the source image.</summary>
/// <param name="Planes">The <c>3 · Width · Height</c> values, channel-major.</param>
/// <param name="Width">Width of the model input in pixels.</param>
/// <param name="Height">Height of the model input in pixels.</param>
/// <param name="SourceWidth">Width of the image that was processed.</param>
/// <param name="SourceHeight">Height of the image that was processed.</param>
/// <param name="ScaleX">Model pixels per source pixel along x (before any crop).</param>
/// <param name="ScaleY">Model pixels per source pixel along y (before any crop).</param>
/// <param name="OffsetX">Left edge of the model input in the scaled image: the crop offset (negative when the image was padded).</param>
/// <param name="OffsetY">Top edge of the model input in the scaled image: the crop offset (negative when the image was padded).</param>
public sealed record ProcessedImage(float[] Planes, int Width, int Height, int SourceWidth, int SourceHeight, float ScaleX, float ScaleY, int OffsetX, int OffsetY)
{
    /// <summary>
    /// Cuts the image into square patches in raster order (patch rows top to bottom, left to right), each flattened channel-major then
    /// row-major: the input of a patch-embedding matrix product (the flattened convolution).
    /// </summary>
    /// <param name="patchSize">Side of a patch; must divide the width and the height.</param>
    /// <returns>A <c>(Width/p · Height/p) · (3·p·p)</c> array.</returns>
    /// <exception cref="ArgumentException">The patch size does not divide the image size.</exception>
    public float[] Patchify(int patchSize)
    {
        if (patchSize <= 0 || Width % patchSize != 0 || Height % patchSize != 0)
        {
            throw new ArgumentException($"Patch size {patchSize} does not divide the {Width}x{Height} model input.");
        }

        int gw = Width / patchSize;
        int gh = Height / patchSize;
        int k = 3 * patchSize * patchSize;
        var patches = new float[(long)gw * gh * k];
        int plane = Width * Height;
        for (int gy = 0; gy < gh; gy++)
        {
            for (int gx = 0; gx < gw; gx++)
            {
                int o = (gy * gw + gx) * k;
                for (int c = 0; c < 3; c++)
                {
                    for (int y = 0; y < patchSize; y++)
                    {
                        Planes.AsSpan(c * plane + (gy * patchSize + y) * Width + gx * patchSize, patchSize)
                            .CopyTo(patches.AsSpan(o + (c * patchSize + y) * patchSize, patchSize));
                    }
                }
            }
        }

        return patches;
    }
}
