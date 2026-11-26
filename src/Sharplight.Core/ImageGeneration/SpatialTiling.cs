namespace Sharplight.ImageGeneration;

/// <summary>
/// Spatial tiling of planar images for networks whose activations do not fit at the full size (VAE encoders and decoders): the input is cut
/// into overlapping tiles, each tile is run on its own, and the outputs are put together with a linear crossfade over the overlaps. Works on
/// any stack of planes ([..., H, W] with the leading dimensions flattened: channels × frames of a latent video, frames × channels of pixels)
/// and on networks that change the resolution by a fixed ratio (a decoder enlarges by its downscale, an encoder reduces by it).
/// </summary>
/// <remarks>
/// The blending is diffusers' (<c>tiled_decode</c> / <c>tiled_encode</c>): tiles are visited row by row; a tile fades in over the tile above
/// across their overlap (vertically first), then over the tile to its left, both of those already blended themselves; each tile then
/// contributes the output from its own start up to the next tile's start. With <see cref="Strided"/> spans this reproduces diffusers exactly;
/// <see cref="Even"/> spans give every tile the same size, so a network compiled per size is compiled once.
/// </remarks>
public static class SpatialTiling
{
    /// <summary>diffusers' layout along one axis: tiles of <paramref name="tile"/> pixels every <paramref name="stride"/> pixels, the last ones shorter.</summary>
    /// <param name="size">The axis length.</param>
    /// <param name="tile">The tile length (positive).</param>
    /// <param name="stride">The distance between tile starts (positive, at most <paramref name="tile"/>).</param>
    /// <returns>The tiles in order.</returns>
    public static IReadOnlyList<TileSpan> Strided(int size, int tile, int stride)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tile);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(stride, tile);
        var spans = new List<TileSpan>();
        for (int start = 0; start < size; start += stride)
        {
            spans.Add(new TileSpan(start, Math.Min(tile, size - start)));
        }

        return spans;
    }

    /// <summary>
    /// <paramref name="count"/> tiles of one length covering an axis, the first at 0 and the last ending at <paramref name="size"/>, neighbours
    /// overlapping by at least <paramref name="overlap"/> pixels; starts and lengths are multiples of <paramref name="multiple"/>.
    /// </summary>
    /// <param name="size">The axis length (a multiple of <paramref name="multiple"/>).</param>
    /// <param name="count">The number of tiles (positive).</param>
    /// <param name="overlap">The least overlap of neighbouring tiles.</param>
    /// <param name="multiple">The alignment of starts and lengths (positive).</param>
    /// <returns>The tiles in order; one tile of the whole axis when <paramref name="count"/> is 1 or the tiles would not be shorter than it.</returns>
    public static IReadOnlyList<TileSpan> Even(int size, int count, int overlap, int multiple)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ArgumentOutOfRangeException.ThrowIfNegative(overlap);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(multiple);
        if (size % multiple != 0)
        {
            throw new ArgumentException($"The axis length {size} is not a multiple of {multiple}.", nameof(size));
        }

        int length = RoundUp((int)Math.Ceiling((size + (count - 1.0) * overlap) / count), multiple);
        if (count == 1 || length >= size)
        {
            return [new TileSpan(0, size)];
        }

        var spans = new TileSpan[count];
        for (int k = 0; k < count; k++)
        {
            int start = (int)((long)k * (size - length) / (count - 1)) / multiple * multiple;
            spans[k] = new TileSpan(k == count - 1 ? size - length : start, length);
        }

        return spans;
    }

    /// <summary>
    /// The fewest <see cref="Even"/> tiles per axis whose area is at most <paramref name="maxTileArea"/> pixels: tiles as square as the
    /// image allows, each side at least <paramref name="minSide"/> (so a tile keeps enough context; an area below that gives the smallest tiles).
    /// </summary>
    /// <param name="height">The image height (a multiple of <paramref name="multiple"/>).</param>
    /// <param name="width">The image width (a multiple of <paramref name="multiple"/>).</param>
    /// <param name="maxTileArea">The largest tile area in pixels (positive).</param>
    /// <param name="overlap">The least overlap of neighbouring tiles.</param>
    /// <param name="multiple">The alignment of starts and lengths.</param>
    /// <param name="minSide">The shortest tile side.</param>
    /// <returns>The row and column tiles.</returns>
    public static (IReadOnlyList<TileSpan> Rows, IReadOnlyList<TileSpan> Columns) Fit(int height, int width, long maxTileArea, int overlap, int multiple, int minSide)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTileArea);
        IReadOnlyList<TileSpan> bestRows = [new TileSpan(0, height)];
        IReadOnlyList<TileSpan> bestColumns = [new TileSpan(0, width)];
        long bestArea = (long)height * width;
        int bestCount = 1;
        int maxRows = Math.Max(1, height / Math.Max(1, minSide));
        int maxColumns = Math.Max(1, width / Math.Max(1, minSide));
        for (int rowCount = 1; rowCount <= maxRows; rowCount++)
        {
            var rows = Even(height, rowCount, overlap, multiple);
            for (int columnCount = 1; columnCount <= maxColumns; columnCount++)
            {
                var columns = Even(width, columnCount, overlap, multiple);
                long area = (long)rows[0].Length * columns[0].Length;
                int count = rows.Count * columns.Count;
                bool fits = area <= maxTileArea;
                bool bestFits = bestArea <= maxTileArea;
                // Prefer a layout that fits, then fewer tiles, then the smaller tile; without any that fits, the smallest tile.
                bool better = fits
                    ? !bestFits || count < bestCount || (count == bestCount && area < bestArea)
                    : !bestFits && area < bestArea;
                if (better)
                {
                    (bestRows, bestColumns, bestArea, bestCount) = (rows, columns, area, count);
                }

                if (fits)
                {
                    break;
                }
            }
        }

        return (bestRows, bestColumns);
    }

    /// <summary>
    /// Runs <paramref name="process"/> on every tile of <paramref name="input"/> and blends the outputs into one image stack.
    /// </summary>
    /// <param name="input">The input: <paramref name="planes"/> planes of <paramref name="height"/> × <paramref name="width"/>.</param>
    /// <param name="planes">Planes of the input.</param>
    /// <param name="height">Input height.</param>
    /// <param name="width">Input width.</param>
    /// <param name="rows">The tiles along the height (from <see cref="Strided"/> or <see cref="Even"/>).</param>
    /// <param name="columns">The tiles along the width.</param>
    /// <param name="scaleNumerator">The output is <paramref name="scaleNumerator"/> / <paramref name="scaleDenominator"/> times the input's size.</param>
    /// <param name="scaleDenominator">See <paramref name="scaleNumerator"/>; every tile boundary must map to whole output pixels.</param>
    /// <param name="blend">The longest crossfade in output pixels (it is also limited by the overlap of the two tiles).</param>
    /// <param name="process">
    /// Runs the network on one tile: its input (<paramref name="planes"/> planes of the tile's height × width) and that height and width;
    /// returns the output planes of the scaled tile size (the same number of planes for every tile).
    /// </param>
    /// <returns>The output: its planes of the scaled height × width.</returns>
    public static float[] Run(float[] input, int planes, int height, int width, IReadOnlyList<TileSpan> rows, IReadOnlyList<TileSpan> columns,
        int scaleNumerator, int scaleDenominator, int blend, Func<float[], int, int, float[]> process)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(process);
        if ((long)planes * height * width != input.Length)
        {
            throw new ArgumentException($"The input holds {input.Length} values, not {planes} planes of {height} × {width}.", nameof(input));
        }

        int Scale(int value)
        {
            long scaled = (long)value * scaleNumerator;
            return scaled % scaleDenominator == 0
                ? (int)(scaled / scaleDenominator)
                : throw new ArgumentException($"A tile boundary at {value} does not map to a whole output pixel at {scaleNumerator}/{scaleDenominator}.");
        }

        int outHeight = Scale(height);
        int outWidth = Scale(width);
        float[]? output = null;
        int outPlanes = 0;
        List<float[]>? above = null;
        for (int i = 0; i < rows.Count; i++)
        {
            var row = new List<float[]>(columns.Count);
            int tileHeight = Scale(rows[i].Length);
            int top = Scale(rows[i].Start);
            int keepHeight = (i + 1 < rows.Count ? Scale(rows[i + 1].Start) : outHeight) - top;
            for (int j = 0; j < columns.Count; j++)
            {
                int tileWidth = Scale(columns[j].Length);
                int left = Scale(columns[j].Start);
                var pixels = process(Cut(input, planes, height, width, rows[i], columns[j]), rows[i].Length, columns[j].Length);
                if (output is null)
                {
                    outPlanes = pixels.Length / (tileHeight * tileWidth);
                    output = new float[(long)outPlanes * outHeight * outWidth];
                }

                if (pixels.Length != outPlanes * tileHeight * tileWidth)
                {
                    throw new InvalidOperationException($"A tile's output holds {pixels.Length} values, not {outPlanes} planes of {tileHeight} × {tileWidth}.");
                }

                if (above is not null)
                {
                    int aboveHeight = Scale(rows[i - 1].Length);
                    int overlap = Scale(rows[i - 1].Start) + aboveHeight - top;
                    BlendVertically(pixels, above[j], outPlanes, tileHeight, tileWidth, aboveHeight, Math.Min(Math.Min(aboveHeight, tileHeight), Math.Min(blend, overlap)));
                }

                if (j > 0)
                {
                    int leftWidth = Scale(columns[j - 1].Length);
                    int overlap = Scale(columns[j - 1].Start) + leftWidth - left;
                    BlendHorizontally(pixels, row[j - 1], outPlanes, tileHeight, tileWidth, leftWidth, Math.Min(Math.Min(leftWidth, tileWidth), Math.Min(blend, overlap)));
                }

                int keepWidth = (j + 1 < columns.Count ? Scale(columns[j + 1].Start) : outWidth) - left;
                Place(pixels, output, outPlanes, tileHeight, tileWidth, Math.Min(keepHeight, tileHeight), Math.Min(keepWidth, tileWidth), top, left, outHeight, outWidth);
                row.Add(pixels);
            }

            above = row;
        }

        return output ?? [];
    }

    private static float[] Cut(float[] input, int planes, int height, int width, TileSpan rows, TileSpan columns)
    {
        if (rows.Start == 0 && columns.Start == 0 && rows.Length == height && columns.Length == width)
        {
            return input;
        }

        var tile = new float[(long)planes * rows.Length * columns.Length];
        for (int p = 0; p < planes; p++)
        {
            for (int y = 0; y < rows.Length; y++)
            {
                input.AsSpan((p * height + rows.Start + y) * width + columns.Start, columns.Length)
                    .CopyTo(tile.AsSpan((p * rows.Length + y) * columns.Length));
            }
        }

        return tile;
    }

    private static void BlendVertically(float[] pixels, float[] above, int planes, int height, int width, int aboveHeight, int extent)
    {
        for (int plane = 0; plane < planes; plane++)
        {
            for (int y = 0; y < extent; y++)
            {
                float weight = (float)y / extent;
                var into = pixels.AsSpan((plane * height + y) * width, width);
                var from = above.AsSpan((plane * aboveHeight + aboveHeight - extent + y) * width, width);
                for (int x = 0; x < width; x++)
                {
                    into[x] = from[x] * (1f - weight) + into[x] * weight;
                }
            }
        }
    }

    private static void BlendHorizontally(float[] pixels, float[] left, int planes, int height, int width, int leftWidth, int extent)
    {
        for (int plane = 0; plane < planes; plane++)
        {
            for (int y = 0; y < height; y++)
            {
                var into = pixels.AsSpan((plane * height + y) * width, width);
                var from = left.AsSpan((plane * height + y) * leftWidth, leftWidth);
                for (int x = 0; x < extent; x++)
                {
                    float weight = (float)x / extent;
                    into[x] = from[leftWidth - extent + x] * (1f - weight) + into[x] * weight;
                }
            }
        }
    }

    private static void Place(float[] pixels, float[] output, int planes, int height, int width, int keepHeight, int keepWidth, int top, int left,
        int outHeight, int outWidth)
    {
        for (int plane = 0; plane < planes; plane++)
        {
            for (int y = 0; y < keepHeight; y++)
            {
                pixels.AsSpan((plane * height + y) * width, keepWidth)
                    .CopyTo(output.AsSpan((plane * outHeight + top + y) * outWidth + left));
            }
        }
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;
}
