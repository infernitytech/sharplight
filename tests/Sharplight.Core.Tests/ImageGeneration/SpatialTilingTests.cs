using Sharplight.ImageGeneration;
using Xunit;

namespace Sharplight.Core.Tests.ImageGeneration;

/// <summary>
/// <see cref="SpatialTiling"/>: tile layouts, an exact reconstruction when the network is a pure resampling, and diffusers' blending
/// reproduced value for value (the algorithm the video generator used before the helper was shared).
/// </summary>
public class SpatialTilingTests
{
    [Fact]
    public void Strided_IsDiffusersLayout()
    {
        Assert.Equal([new TileSpan(0, 64), new TileSpan(48, 64), new TileSpan(96, 34)], SpatialTiling.Strided(130, 64, 48));
    }

    [Theory]
    [InlineData(480, 1, 64, 16)]
    [InlineData(480, 2, 64, 16)]
    [InlineData(1088, 3, 64, 16)]
    [InlineData(1920, 5, 64, 16)]
    [InlineData(136, 4, 8, 2)]
    public void Even_CoversTheAxisWithEqualAlignedOverlappingTiles(int size, int count, int overlap, int multiple)
    {
        var spans = SpatialTiling.Even(size, count, overlap, multiple);
        Assert.Equal(0, spans[0].Start);
        Assert.Equal(size, spans[^1].Start + spans[^1].Length);
        Assert.All(spans, span => Assert.Equal(spans[0].Length, span.Length));
        Assert.All(spans, span => Assert.True(span.Start % multiple == 0 && span.Length % multiple == 0, $"{span}"));
        for (int k = 1; k < spans.Count; k++)
        {
            Assert.True(spans[k - 1].Start + spans[k - 1].Length - spans[k].Start >= overlap, $"{spans[k - 1]} {spans[k]}");
        }
    }

    [Fact]
    public void Fit_PrefersFewTilesThatFitTheArea()
    {
        var (rows, columns) = SpatialTiling.Fit(1088, 1920, 1088L * 1920, 64, 16, 128);
        Assert.Single(rows);
        Assert.Single(columns);

        (rows, columns) = SpatialTiling.Fit(1088, 1920, 700_000, 64, 16, 128);
        Assert.True((long)rows[0].Length * columns[0].Length <= 700_000);
        Assert.True(rows.Count * columns.Count <= 4, $"{rows.Count} x {columns.Count}");

        // Nothing fits: the smallest tiles the shortest side allows.
        (rows, columns) = SpatialTiling.Fit(256, 256, 10, 64, 16, 128);
        Assert.Equal(2, rows.Count);
        Assert.Equal(2, columns.Count);
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(8, 1)]
    public void Run_OfAResampling_ReconstructsTheWholeImage(int numerator, int denominator)
    {
        // A "network" that enlarges by nearest neighbour (or keeps every 8th pixel): tiles of it agree on their overlaps, so blending is exact.
        const int planes = 6;
        const int height = 96;
        const int width = 160;
        var input = new float[planes * height * width];
        var random = new Random(3);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)random.NextDouble();
        }

        float[] Resample(float[] image, int h, int w)
        {
            int outH = h * numerator / denominator;
            int outW = w * numerator / denominator;
            var output = new float[planes * outH * outW];
            for (int p = 0; p < planes; p++)
            {
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        output[(p * outH + y) * outW + x] = image[(p * h + y * denominator / numerator) * w + x * denominator / numerator];
                    }
                }
            }

            return output;
        }

        var whole = Resample(input, height, width);
        var tiled = SpatialTiling.Run(input, planes, height, width, SpatialTiling.Even(height, 2, 16, 8), SpatialTiling.Even(width, 3, 16, 8),
            numerator, denominator, int.MaxValue, Resample);
        Assert.Equal(whole.Length, tiled.Length);
        // Equal up to the rounding of a · (1 − w) + a · w.
        Assert.All(Enumerable.Range(0, whole.Length), i => Assert.Equal(whole[i], tiled[i], 1e-6f));
    }

    [Fact]
    public void Run_WithStridedTiles_IsDiffusersBlending()
    {
        const int planes = 2;
        const int height = 21;
        const int width = 27;
        const int tile = 8;
        const int stride = 6;
        const int scale = 2;
        var input = new float[planes * height * width];
        var random = new Random(5);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)random.NextDouble();
        }

        // Each tile's "network": enlarge and add a value that depends on the tile, so the overlaps differ and the blending shows.
        int calls = 0;
        float[] Network(float[] image, int h, int w)
        {
            float offset = 10 * calls++;
            var output = new float[planes * h * scale * w * scale];
            for (int p = 0; p < planes; p++)
            {
                for (int y = 0; y < h * scale; y++)
                {
                    for (int x = 0; x < w * scale; x++)
                    {
                        output[(p * h * scale + y) * w * scale + x] = image[(p * h + y / scale) * w + x / scale] + offset;
                    }
                }
            }

            return output;
        }

        var tiled = SpatialTiling.Run(input, planes, height, width, SpatialTiling.Strided(height, tile, stride), SpatialTiling.Strided(width, tile, stride),
            scale, 1, (tile - stride) * scale, Network);
        calls = 0;
        var reference = DiffusersReference(input, planes, height, width, tile, stride, scale, Network);
        Assert.Equal(reference, tiled);
    }

    /// <summary>The video generator's spatial tiling as it was before <see cref="SpatialTiling"/> (diffusers' <c>tiled_decode</c>).</summary>
    private static float[] DiffusersReference(float[] latents, int planes, int latentHeight, int latentWidth, int tileSize, int strideLatent, int down,
        Func<float[], int, int, float[]> decode)
    {
        int blend = (tileSize - strideLatent) * down;
        int strideOut = strideLatent * down;
        int outHeight = latentHeight * down;
        int outWidth = latentWidth * down;
        var rows = new List<List<(float[] Pixels, int Height, int Width)>>();
        for (int top = 0; top < latentHeight; top += strideLatent)
        {
            var row = new List<(float[], int, int)>();
            for (int left = 0; left < latentWidth; left += strideLatent)
            {
                int height = Math.Min(tileSize, latentHeight - top);
                int width = Math.Min(tileSize, latentWidth - left);
                var tile = new float[planes * height * width];
                for (int p = 0; p < planes; p++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        latents.AsSpan((p * latentHeight + top + y) * latentWidth + left, width).CopyTo(tile.AsSpan((p * height + y) * width));
                    }
                }

                row.Add((decode(tile, height, width), height * down, width * down));
            }

            rows.Add(row);
        }

        var output = new float[planes * outHeight * outWidth];
        int outTop = 0;
        for (int i = 0; i < rows.Count; i++)
        {
            int outLeft = 0;
            int keptHeight = 0;
            for (int j = 0; j < rows[i].Count; j++)
            {
                var (pixels, height, width) = rows[i][j];
                if (i > 0)
                {
                    var (above, aboveHeight, _) = rows[i - 1][j];
                    int extent = Math.Min(Math.Min(aboveHeight, height), blend);
                    for (int image = 0; image < planes; image++)
                    {
                        for (int y = 0; y < extent; y++)
                        {
                            float weight = (float)y / extent;
                            for (int x = 0; x < width; x++)
                            {
                                int into = (image * height + y) * width + x;
                                pixels[into] = above[(image * aboveHeight + aboveHeight - extent + y) * width + x] * (1f - weight) + pixels[into] * weight;
                            }
                        }
                    }
                }

                if (j > 0)
                {
                    var (leftTile, _, leftWidth) = rows[i][j - 1];
                    int extent = Math.Min(Math.Min(leftWidth, width), blend);
                    for (int image = 0; image < planes; image++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < extent; x++)
                            {
                                float weight = (float)x / extent;
                                int into = (image * height + y) * width + x;
                                pixels[into] = leftTile[(image * height + y) * leftWidth + leftWidth - extent + x] * (1f - weight) + pixels[into] * weight;
                            }
                        }
                    }
                }

                int keepHeight = Math.Min(Math.Min(height, strideOut), outHeight - outTop);
                int keepWidth = Math.Min(Math.Min(width, strideOut), outWidth - outLeft);
                for (int image = 0; image < planes; image++)
                {
                    for (int y = 0; y < keepHeight; y++)
                    {
                        pixels.AsSpan((image * height + y) * width, keepWidth).CopyTo(output.AsSpan((image * outHeight + outTop + y) * outWidth + outLeft));
                    }
                }

                outLeft += keepWidth;
                keptHeight = keepHeight;
            }

            outTop += keptHeight;
        }

        return output;
    }
}
