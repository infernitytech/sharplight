namespace Sharplight.Vision;

/// <summary>A decoded image: 8-bit interleaved RGB (row-major, 3 bytes per pixel). File decoding is the host's job.</summary>
public sealed record ImageInput(int Width, int Height, byte[] Rgb)
{
    public void Validate()
    {
        if (Width <= 0 || Height <= 0) throw new ArgumentException("Image has no pixels.");
        if (Rgb.LongLength != 3L * Width * Height) throw new ArgumentException($"Expected {3L * Width * Height} RGB bytes for {Width}x{Height}, got {Rgb.Length}.");
    }
}

/// <summary>Optional limits on the visual token budget per image (output tokens after merging), overriding the model's pixel bounds.</summary>
public sealed record ImageTokenLimits(int? MinTokens = null, int? MaxTokens = null);

/// <summary>An image converted to the inputs of the vision-encode graph (see <see cref="VisionParameters"/>).</summary>
public sealed class PreprocessedImage
{
    /// <summary>Size of the resized image in pixels.</summary>
    public required int Width { get; init; }
    public required int Height { get; init; }
    /// <summary>Patch grid (rows × columns of patches).</summary>
    public required int GridHeight { get; init; }
    public required int GridWidth { get; init; }
    /// <summary>Output token grid after spatial merging.</summary>
    public required int MergedHeight { get; init; }
    public required int MergedWidth { get; init; }
    /// <summary>Normalized patch values [N, PatchDim], merge-window order, each patch flattened in (channel, frame, row, col) order.</summary>
    public required float[] Patches { get; init; }
    /// <summary>(row, col) of each patch in the patch grid [N, 2].</summary>
    public required int[] Positions { get; init; }
    /// <summary>Bilinear taps into the learned position grid [N, 4] (row-major grid index) and their weights [N, 4].</summary>
    public required int[] EmbeddingIndex { get; init; }
    public required float[] EmbeddingWeight { get; init; }

    public int PatchCount => GridHeight * GridWidth;
    /// <summary>Number of embeddings the encoder produces (and language-model positions the image occupies as rows).</summary>
    public int TokenCount => MergedHeight * MergedWidth;
}

/// <summary>
/// Turns raw RGB pixels into the inputs of a model's vision encoder, driven only by the canonical
/// <see cref="VisionParameters"/>: aspect-preserving "smart" resize to a multiple of patch_size·merge_size within the
/// pixel bounds (antialiased bicubic), per-channel normalization, patchification (the still image is repeated to fill
/// temporal_patch_size frames) in spatial-merge-window order, plus the position inputs (row/col per patch and the
/// bilinear taps into the learned position grid, align_corners semantics).
/// </summary>
public sealed class VisionPreprocessor(VisionParameters p, ImageTokenLimits? limits = null)
{
    public VisionParameters Parameters => p;

    /// <summary>Target size in pixels for an input size (smart resize).</summary>
    public (int Width, int Height) TargetSize(int width, int height)
    {
        int factor = p.PatchSize * p.SpatialMergeSize;
        long unit = (long)factor * factor;
        long minPixels = limits?.MinTokens is int mt ? mt * unit : p.MinPixels;
        long maxPixels = limits?.MaxTokens is int xt ? xt * unit : p.MaxPixels;
        double h = height, w = width;
        // Python round() semantics (half to even), as the reference processors.
        int hBar = Math.Max(factor, (int)Math.Round(h / factor, MidpointRounding.ToEven) * factor);
        int wBar = Math.Max(factor, (int)Math.Round(w / factor, MidpointRounding.ToEven) * factor);
        if (maxPixels > 0 && (long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt(h * w / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(h / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(w / beta / factor) * factor);
        }
        else if (minPixels > 0 && (long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / (h * w));
            hBar = (int)Math.Ceiling(h * beta / factor) * factor;
            wBar = (int)Math.Ceiling(w * beta / factor) * factor;
        }
        return (wBar, hBar);
    }

    public PreprocessedImage Process(ImageInput image)
    {
        image.Validate();
        if (p.InChannels != 3) throw new NotSupportedException($"Only 3-channel images are supported (in_channels {p.InChannels}).");
        var (tw, th) = TargetSize(image.Width, image.Height);
        byte[] px = image.Width == tw && image.Height == th ? image.Rgb : ResizeBicubic(image.Rgb, image.Width, image.Height, tw, th);

        int ps = p.PatchSize, m = p.SpatialMergeSize, tp = p.TemporalPatchSize;
        int gh = th / ps, gw = tw / ps, n = gh * gw, k = p.PatchDim;
        var patches = new float[checked(n * k)];
        var positions = new int[n * 2];
        // Normalization: (x/255 - mean) / std  ==  x * a + b.
        Span<float> a = stackalloc float[3], b = stackalloc float[3];
        for (int c = 0; c < 3; c++) { a[c] = 1f / (255f * p.ImageStd[c]); b[c] = -p.ImageMean[c] / p.ImageStd[c]; }

        int idx = 0;
        for (int br = 0; br < gh / m; br++)
            for (int bc = 0; bc < gw / m; bc++)
                for (int ir = 0; ir < m; ir++)
                    for (int ic = 0; ic < m; ic++, idx++)
                    {
                        int row = br * m + ir, col = bc * m + ic;
                        positions[idx * 2] = row; positions[idx * 2 + 1] = col;
                        var dst = patches.AsSpan(checked(idx * k), k);
                        int o = 0;
                        for (int c = 0; c < 3; c++)
                            for (int t = 0; t < tp; t++)
                                for (int y = 0; y < ps; y++)
                                {
                                    int src = ((row * ps + y) * tw + col * ps) * 3 + c;
                                    for (int x = 0; x < ps; x++, src += 3) dst[o++] = px[src] * a[c] + b[c];
                                }
                    }

        var (ei, ew) = PositionTaps(positions, gh, gw);
        return new PreprocessedImage
        {
            Width = tw, Height = th, GridHeight = gh, GridWidth = gw, MergedHeight = gh / m, MergedWidth = gw / m,
            Patches = patches, Positions = positions, EmbeddingIndex = ei, EmbeddingWeight = ew,
        };
    }

    /// <summary>Bilinear taps (align_corners) resampling the learned g×g position grid to the image grid, per patch.</summary>
    (int[] Index, float[] Weight) PositionTaps(int[] positions, int gh, int gw)
    {
        int n = positions.Length / 2, side = Math.Max(1, p.PositionGridSide);
        var index = new int[n * 4];
        var weight = new float[n * 4];
        if (p.PositionEmbeddingCount == 0) return (index, weight);
        static void Axis(int pos, int count, int side, out int i0, out int i1, out float w0, out float w1)
        {
            double f = count > 1 ? pos * (side - 1.0) / (count - 1) : 0;
            i0 = Math.Min((int)Math.Floor(f), side - 1);
            i1 = Math.Min(i0 + 1, side - 1);
            w1 = (float)(f - i0); w0 = 1f - w1;
        }
        for (int i = 0; i < n; i++)
        {
            Axis(positions[i * 2], gh, side, out int r0, out int r1, out float wr0, out float wr1);
            Axis(positions[i * 2 + 1], gw, side, out int c0, out int c1, out float wc0, out float wc1);
            index[i * 4 + 0] = r0 * side + c0; weight[i * 4 + 0] = wr0 * wc0;
            index[i * 4 + 1] = r0 * side + c1; weight[i * 4 + 1] = wr0 * wc1;
            index[i * 4 + 2] = r1 * side + c0; weight[i * 4 + 2] = wr1 * wc0;
            index[i * 4 + 3] = r1 * side + c1; weight[i * 4 + 3] = wr1 * wc1;
        }
        return (index, weight);
    }

    // ---- antialiased bicubic resize (PIL / torch antialias=True semantics, Keys kernel a = -0.5) ----------------------
    static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
    }

    static (int[] Start, int[] Count, float[] Weights, int Max) Coefficients(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = 2.0 * filterScale;
        int max = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[outSize]; var count = new int[outSize]; var w = new float[outSize * max];
        for (int x = 0; x < outSize; x++)
        {
            double center = (x + 0.5) * scale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize);
            double sum = 0;
            var tmp = new double[xmax - xmin];
            for (int i = 0; i < tmp.Length; i++) { tmp[i] = Cubic((xmin + i - center + 0.5) / filterScale); sum += tmp[i]; }
            start[x] = xmin; count[x] = tmp.Length;
            for (int i = 0; i < tmp.Length; i++) w[x * max + i] = (float)(tmp[i] / sum);
        }
        return (start, count, w, max);
    }

    /// <summary>Antialiased bicubic resize of interleaved 8-bit RGB (rounded and clamped to 8 bits like the reference).</summary>
    public static byte[] ResizeBicubic(byte[] src, int sw, int sh, int dw, int dh)
    {
        var (hs, hc, hw, hm) = Coefficients(sw, dw);
        var (vs, vc, vw, vm) = Coefficients(sh, dh);
        // Horizontal pass keeps float precision, vertical pass rounds.
        var tmp = new float[(long)sh * dw * 3];
        Parallel.For(0, sh, y =>
        {
            for (int x = 0; x < dw; x++)
            {
                float r = 0, g = 0, b = 0;
                for (int i = 0; i < hc[x]; i++)
                {
                    float w = hw[x * hm + i];
                    int s = (y * sw + hs[x] + i) * 3;
                    r += src[s] * w; g += src[s + 1] * w; b += src[s + 2] * w;
                }
                int o = (y * dw + x) * 3;
                tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
            }
        });
        var dst = new byte[(long)dw * dh * 3];
        Parallel.For(0, dh, y =>
        {
            for (int x = 0; x < dw; x++)
            {
                float r = 0, g = 0, b = 0;
                for (int i = 0; i < vc[y]; i++)
                {
                    float w = vw[y * vm + i];
                    int s = ((vs[y] + i) * dw + x) * 3;
                    r += tmp[s] * w; g += tmp[s + 1] * w; b += tmp[s + 2] * w;
                }
                int o = (y * dw + x) * 3;
                dst[o] = Clamp8(r); dst[o + 1] = Clamp8(g); dst[o + 2] = Clamp8(b);
            }
        });
        return dst;
    }

    static byte Clamp8(float v) => (byte)Math.Clamp((int)MathF.Round(v, MidpointRounding.AwayFromZero), 0, 255);
}
