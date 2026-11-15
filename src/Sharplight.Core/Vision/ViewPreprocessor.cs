namespace Sharplight.Vision;

/// <summary>
/// Image preprocessing of two-stage towers of fixed-size views (<see cref="SamParameters"/>): an image larger than a tile is also cut into a grid
/// of tiles (the grid of <see cref="SamParameters.MinTiles"/> to <see cref="SamParameters.MaxTiles"/> tiles whose aspect ratio is closest to the
/// image's, the image resized to fill it); the global view is the image resized to fit <see cref="SamParameters.GlobalSize"/> and padded to a
/// square with <see cref="SamParameters.PadValue"/>. Resizing is antialiased bicubic on 8-bit pixels, then the views are normalized.
/// </summary>
internal static class ViewPreprocessor
{
    /// <summary>Preprocesses a decoded image.</summary>
    public static PreprocessedImage Process(VisionParameters p, ImageInput image)
    {
        var sam = p.Sam!;
        int G = sam.GlobalSize;
        int T = sam.TileSize;
        var tiles = new List<float[]>();
        int columns = 0;
        int rows = 0;
        if (Math.Max(image.Width, image.Height) > T)
        {
            (columns, rows) = TileGrid(image.Width, image.Height, T, sam.MinTiles, sam.MaxTiles);
            int cw = columns * T;
            int ch = rows * T;
            var canvas = image.Width == cw && image.Height == ch ? image.Rgb : VisionPreprocessor.ResizeBicubic(image.Rgb, image.Width, image.Height, cw, ch);
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    tiles.Add(Normalize(p, canvas, cw, c * T, r * T, T));
                }
            }
        }

        double scale = (double)G / Math.Max(image.Width, image.Height);
        int nw = (int)Math.Round(image.Width * scale, MidpointRounding.ToEven);
        int nh = (int)Math.Round(image.Height * scale, MidpointRounding.ToEven);
        var resized = image.Width == nw && image.Height == nh ? image.Rgb : VisionPreprocessor.ResizeBicubic(image.Rgb, image.Width, image.Height, nw, nh);
        var square = new byte[3 * G * G];
        Array.Fill(square, (byte)sam.PadValue);
        int left = (G - nw) / 2;
        int top = (G - nh) / 2;
        for (int y = 0; y < nh; y++)
        {
            Array.Copy(resized, 3 * y * nw, square, 3 * ((top + y) * G + left), 3 * nw);
        }

        return Views(p, Normalize(p, square, G, 0, 0, G), tiles, columns, rows);
    }

    /// <summary>
    /// Builds the encoder input from already normalized channels-first views: the global view [3, G, G] and the tiles [3, T, T] in raster order
    /// of a <paramref name="columns"/> × <paramref name="rows"/> grid (none: 0 × 0).
    /// </summary>
    public static PreprocessedImage Views(VisionParameters p, float[] global, IReadOnlyList<float[]> tiles, int columns, int rows)
    {
        var sam = p.Sam!;
        int G = sam.GlobalSize;
        int T = sam.TileSize;
        if (global.Length != 3 * G * G || tiles.Count != columns * rows || tiles.Any(t => t.Length != 3 * T * T))
        {
            throw new ArgumentException($"Views must be a {G}×{G} global view and {columns}×{rows} tiles of {T}×{T}.");
        }

        int f = sam.FeatureSide(G, p.PatchSize);
        int nl = sam.Newlines ? 1 : 0;
        int tokens = f * (f + nl) + 1;
        if (tiles.Count > 0)
        {
            int ft = sam.FeatureSide(T, p.PatchSize);
            tokens += rows * ft * (columns * ft + nl);
        }

        var pixels = new float[global.Length + tiles.Count * 3 * T * T];
        global.CopyTo(pixels, 0);
        for (int i = 0; i < tiles.Count; i++)
        {
            tiles[i].CopyTo(pixels, global.Length + i * 3 * T * T);
        }

        int g = G / p.PatchSize;
        return new PreprocessedImage
        {
            Width = G, Height = G, GridHeight = g, GridWidth = g, MergedHeight = 1, MergedWidth = tokens,
            Patches = pixels, Positions = [], EmbeddingIndex = [], EmbeddingWeight = [], TileColumns = columns, TileRows = rows,
        };
    }

    /// <summary>
    /// The (columns, rows) tile grid with the aspect ratio closest to the image's among grids of <paramref name="min"/> to <paramref name="max"/>
    /// tiles (ordered by tile count); on a tie the later grid wins while the image covers more than half of its area.
    /// </summary>
    internal static (int Columns, int Rows) TileGrid(int width, int height, int tile, int min, int max)
    {
        var grids = new List<(int W, int H)>();
        for (int w = 1; w <= max; w++)
        {
            for (int h = 1; h <= max; h++)
            {
                if (w * h <= max && w * h >= min)
                {
                    grids.Add((w, h));
                }
            }
        }

        // A stable sort by tile count, as Python's sorted.
        grids = [.. grids.Select((g, i) => (g, i)).OrderBy(x => x.g.W * x.g.H).ThenBy(x => x.i).Select(x => x.g)];
        double aspect = (double)width / height;
        double area = (double)width * height;
        double best = double.PositiveInfinity;
        var result = (1, 1);
        foreach (var (w, h) in grids)
        {
            double diff = Math.Abs(aspect - (double)w / h);
            if (diff < best)
            {
                best = diff;
                result = (w, h);
            }
            else if (diff == best && area > 0.5 * tile * tile * w * h)
            {
                result = (w, h);
            }
        }

        return result;
    }

    /// <summary>A size × size crop at (x0, y0) of interleaved 8-bit RGB (row stride <paramref name="width"/>), normalized, channels first.</summary>
    private static float[] Normalize(VisionParameters p, byte[] rgb, int width, int x0, int y0, int size)
    {
        var v = new float[3 * size * size];
        for (int c = 0; c < 3; c++)
        {
            float a = 1f / (255f * p.ImageStd[c]);
            float b = -p.ImageMean[c] / p.ImageStd[c];
            for (int y = 0; y < size; y++)
            {
                int src = ((y0 + y) * width + x0) * 3 + c;
                int dst = (c * size + y) * size;
                for (int x = 0; x < size; x++, src += 3)
                {
                    v[dst + x] = rgb[src] * a + b;
                }
            }
        }

        return v;
    }
}
