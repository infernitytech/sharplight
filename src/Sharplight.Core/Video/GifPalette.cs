namespace Sharplight.Video;

/// <summary>
/// A palette of up to 256 colors for one frame and the mapping of the frame's pixels to it. Frames with at most 256 distinct colors get an
/// exact palette; others are quantized with median cut over a 5-bit-per-channel histogram, optionally with 4x4 ordered dithering.
/// </summary>
internal sealed class GifPalette
{
    private static readonly int[] s_bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

    private readonly byte[] _colors;
    private readonly int _count;

    private GifPalette(byte[] colors, int count, bool exact)
    {
        _colors = colors;
        _count = count;
        IsExact = exact;
    }

    /// <summary>Whether every pixel of the frame is represented by its exact color.</summary>
    public bool IsExact { get; }

    /// <summary>Bits needed to index the palette (1..8).</summary>
    public int BitsPerPixel
    {
        get
        {
            int bits = 1;
            while ((1 << bits) < _count)
            {
                bits++;
            }

            return bits;
        }
    }

    /// <summary>The palette padded with black to <c>2^BitsPerPixel</c> entries, RGB triplets.</summary>
    public byte[] Table
    {
        get
        {
            var table = new byte[3 << BitsPerPixel];
            _colors.AsSpan(0, _count * 3).CopyTo(table);
            return table;
        }
    }

    /// <summary>Builds the palette for a frame.</summary>
    public static GifPalette Build(ReadOnlySpan<byte> rgb)
    {
        var exact = new Dictionary<int, int>();
        var exactColors = new List<byte>();
        bool fits = true;
        for (int i = 0; i < rgb.Length; i += 3)
        {
            int key = (rgb[i] << 16) | (rgb[i + 1] << 8) | rgb[i + 2];
            if (!exact.ContainsKey(key))
            {
                if (exact.Count == 256)
                {
                    fits = false;
                    break;
                }

                exact[key] = exact.Count;
                exactColors.Add(rgb[i]);
                exactColors.Add(rgb[i + 1]);
                exactColors.Add(rgb[i + 2]);
            }
        }

        if (fits)
        {
            return new GifPalette([.. exactColors], exact.Count, exact: true);
        }

        return Quantize(rgb);
    }

    /// <summary>Maps the pixels of a frame to palette indices.</summary>
    /// <param name="rgb">Interleaved RGB.</param>
    /// <param name="width">Frame width, for the dither pattern.</param>
    /// <param name="dither">Applies ordered dithering (ignored for exact palettes).</param>
    public byte[] Map(ReadOnlySpan<byte> rgb, int width, bool dither)
    {
        var indices = new byte[rgb.Length / 3];
        if (IsExact)
        {
            var lookup = new Dictionary<int, byte>(_count);
            for (int c = 0; c < _count; c++)
            {
                lookup[(_colors[c * 3] << 16) | (_colors[c * 3 + 1] << 8) | _colors[c * 3 + 2]] = (byte)c;
            }

            for (int p = 0; p < indices.Length; p++)
            {
                indices[p] = lookup[(rgb[p * 3] << 16) | (rgb[p * 3 + 1] << 8) | rgb[p * 3 + 2]];
            }

            return indices;
        }

        var cache = new short[1 << 15];
        Array.Fill(cache, (short)-1);
        for (int p = 0; p < indices.Length; p++)
        {
            int r = rgb[p * 3];
            int g = rgb[p * 3 + 1];
            int b = rgb[p * 3 + 2];
            if (dither)
            {
                int bias = s_bayer[((p / width) & 3) * 4 + ((p % width) & 3)] - 8;
                r = Math.Clamp(r + bias, 0, 255);
                g = Math.Clamp(g + bias, 0, 255);
                b = Math.Clamp(b + bias, 0, 255);
            }

            int key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
            short found = cache[key];
            if (found < 0)
            {
                found = (short)Nearest((key >> 10 << 3) | 4, ((key >> 5) & 31) << 3 | 4, (key & 31) << 3 | 4);
                cache[key] = found;
            }

            indices[p] = (byte)found;
        }

        return indices;
    }

    private static GifPalette Quantize(ReadOnlySpan<byte> rgb)
    {
        // Histogram over 15-bit colors; each bucket remembers the sums of its exact colors.
        var counts = new int[1 << 15];
        var sums = new long[(1 << 15) * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            int key = ((rgb[i] >> 3) << 10) | ((rgb[i + 1] >> 3) << 5) | (rgb[i + 2] >> 3);
            counts[key]++;
            sums[key * 3] += rgb[i];
            sums[key * 3 + 1] += rgb[i + 1];
            sums[key * 3 + 2] += rgb[i + 2];
        }

        var all = new List<int>();
        for (int key = 0; key < counts.Length; key++)
        {
            if (counts[key] > 0)
            {
                all.Add(key);
            }
        }

        var boxes = new List<List<int>> { all };
        while (boxes.Count < 256)
        {
            int target = -1;
            long best = 0;
            for (int b = 0; b < boxes.Count; b++)
            {
                if (boxes[b].Count < 2)
                {
                    continue;
                }

                long weight = BoxWeight(boxes[b], counts, out _);
                if (weight > best)
                {
                    best = weight;
                    target = b;
                }
            }

            if (target < 0)
            {
                break;
            }

            List<int> box = boxes[target];
            BoxWeight(box, counts, out int axis);
            int shift = (2 - axis) * 5;
            box.Sort((a, b) => ((a >> shift) & 31).CompareTo((b >> shift) & 31));
            long total = 0;
            foreach (int key in box)
            {
                total += counts[key];
            }

            long running = 0;
            int split = 1;
            for (int i = 0; i < box.Count - 1; i++)
            {
                running += counts[box[i]];
                split = i + 1;
                if (running * 2 >= total)
                {
                    break;
                }
            }

            boxes[target] = box.GetRange(0, split);
            boxes.Add(box.GetRange(split, box.Count - split));
        }

        var colors = new byte[boxes.Count * 3];
        for (int b = 0; b < boxes.Count; b++)
        {
            long n = 0;
            long r = 0;
            long g = 0;
            long bl = 0;
            foreach (int key in boxes[b])
            {
                n += counts[key];
                r += sums[key * 3];
                g += sums[key * 3 + 1];
                bl += sums[key * 3 + 2];
            }

            colors[b * 3] = (byte)((r + n / 2) / n);
            colors[b * 3 + 1] = (byte)((g + n / 2) / n);
            colors[b * 3 + 2] = (byte)((bl + n / 2) / n);
        }

        return new GifPalette(colors, boxes.Count, exact: false);
    }

    /// <summary>The population-weighted extent of a box and the channel (0 red, 1 green, 2 blue) it is widest in.</summary>
    private static long BoxWeight(List<int> box, int[] counts, out int axis)
    {
        Span<int> min = [31, 31, 31];
        Span<int> max = [0, 0, 0];
        long total = 0;
        foreach (int key in box)
        {
            total += counts[key];
            for (int c = 0; c < 3; c++)
            {
                int value = (key >> ((2 - c) * 5)) & 31;
                min[c] = Math.Min(min[c], value);
                max[c] = Math.Max(max[c], value);
            }
        }

        axis = 0;
        int widest = max[0] - min[0];
        for (int c = 1; c < 3; c++)
        {
            if (max[c] - min[c] > widest)
            {
                widest = max[c] - min[c];
                axis = c;
            }
        }

        return total * (widest + 1);
    }

    private int Nearest(int r, int g, int b)
    {
        int best = 0;
        int bestDistance = int.MaxValue;
        for (int c = 0; c < _count; c++)
        {
            int dr = _colors[c * 3] - r;
            int dg = _colors[c * 3 + 1] - g;
            int db = _colors[c * 3 + 2] - b;
            int distance = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = c;
            }
        }

        return best;
    }
}
