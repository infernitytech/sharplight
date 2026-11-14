namespace Sharplight.Vision;

/// <summary>
/// Turns raw RGB pixels into the inputs of a model's vision encoder, driven only by the canonical
/// <see cref="VisionParameters"/>: aspect-preserving "smart" resize to a multiple of patch_size·merge_size within the
/// pixel bounds (antialiased bicubic), per-channel normalization, patchification (the still image is repeated to fill
/// temporal_patch_size frames) in spatial-merge-window order (window by window when the tower has windowed attention), plus the position inputs (row/col per patch and the
/// bilinear taps into the learned position grid, align_corners semantics).
/// </summary>
public sealed class VisionPreprocessor(VisionParameters p, ImageTokenLimits? limits = null)
{
    /// <summary>The vision tower parameters this preprocessor follows.</summary>
    public VisionParameters Parameters => p;

    /// <summary>Target size in pixels for an input size (smart resize).</summary>
    internal (int Width, int Height) TargetSize(int width, int height)
    {
        int factor = p.PatchSize * p.SpatialMergeSize;
        long unit = (long)factor * factor;
        long minPixels = limits?.MinTokens is int mt ? mt * unit : p.MinPixels;
        long maxPixels = limits?.MaxTokens is int xt ? xt * unit : p.MaxPixels;
        double h = height;
        double w = width;
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

    /// <summary>
    /// Converts an image into encoder inputs. The input image is not modified.
    /// </summary>
    /// <param name="image">The decoded image.</param>
    /// <exception cref="ArgumentException">The image has no pixels or its byte count does not match its size.</exception>
    /// <exception cref="NotSupportedException">The tower does not take 3-channel input.</exception>
    public PreprocessedImage Process(ImageInput image)
    {
        image.Validate();
        if (p.InChannels != 3)
        {
            throw new NotSupportedException($"Only 3-channel images are supported (in_channels {p.InChannels}).");
        }

        var (tw, th) = TargetSize(image.Width, image.Height);
        byte[] px = image.Width == tw && image.Height == th ? image.Rgb : ResizeBicubic(image.Rgb, image.Width, image.Height, tw, th);

        // Normalization: (x/255 - mean) / std  ==  x * a + b.
        Span<float> a = stackalloc float[3];
        Span<float> b = stackalloc float[3];
        for (int c = 0; c < 3; c++)
        {
            a[c] = 1f / (255f * p.ImageStd[c]);
            b[c] = -p.ImageMean[c] / p.ImageStd[c];
        }
        var values = new float[px.Length];
        for (int i = 0; i < px.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
            {
                values[i + c] = px[i + c] * a[c] + b[c];
            }
        }

        return Patchify(values, tw, th);
    }

    /// <summary>
    /// Patchifies an already normalized image (interleaved RGB floats, width and height multiples of patch_size·merge_size)
    /// into the encoder inputs; what <see cref="Process"/> does after resizing and normalizing.
    /// </summary>
    internal PreprocessedImage Patchify(float[] rgb, int tw, int th)
    {
        int ps = p.PatchSize;
        int m = p.SpatialMergeSize;
        int tp = p.TemporalPatchSize;
        int gh = th / ps;
        int gw = tw / ps;
        int n = gh * gw;
        int k = p.PatchDim;
        if (th % (ps * m) != 0 || tw % (ps * m) != 0 || rgb.Length != 3 * tw * th)
        {
            throw new ArgumentException($"Image {tw}x{th} does not fit patch_size {ps} x spatial_merge_size {m}.");
        }

        var patches = new float[checked(n * k)];
        var positions = new int[n * 2];
        var (units, spans, order) = MergeUnitOrder(gh / m, gw / m);
        int unitPatches = m * m;
        for (int u = 0; u < units.Length; u++)
        {
            int br = units[u] / (gw / m);
            int bc = units[u] % (gw / m);
            for (int ir = 0; ir < m; ir++)
            {
                for (int ic = 0; ic < m; ic++)
                {
                    int idx = u * unitPatches + ir * m + ic;
                    int row = br * m + ir;
                    int col = bc * m + ic;
                    positions[idx * 2] = row;
                    positions[idx * 2 + 1] = col;
                    var dst = patches.AsSpan(checked(idx * k), k);
                    int o = 0;
                    for (int c = 0; c < 3; c++)
                    {
                        for (int t = 0; t < tp; t++)
                        {
                            for (int y = 0; y < ps; y++)
                            {
                                int src = ((row * ps + y) * tw + col * ps) * 3 + c;
                                for (int x = 0; x < ps; x++, src += 3)
                                {
                                    dst[o++] = rgb[src];
                                }
                            }
                        }
                    }
                }
            }
        }

        var patchSpans = Array.Empty<int>();
        if (spans.Length > 0)
        {
            patchSpans = new int[n * 2];
            for (int u = 0; u < units.Length; u++)
            {
                for (int j = 0; j < unitPatches; j++)
                {
                    patchSpans[(u * unitPatches + j) * 2] = spans[u * 2] * unitPatches;
                    patchSpans[(u * unitPatches + j) * 2 + 1] = spans[u * 2 + 1] * unitPatches;
                }
            }
        }

        var (ei, ew) = PositionTaps(positions, gh, gw);
        return new PreprocessedImage
        {
            Width = tw, Height = th, GridHeight = gh, GridWidth = gw, MergedHeight = gh / m, MergedWidth = gw / m,
            Patches = patches, Positions = positions, EmbeddingIndex = ei, EmbeddingWeight = ew, WindowSpans = patchSpans, WindowOrder = order,
        };
    }

    /// <summary>
    /// Order of the merged tokens (merge units) of a mh × mw grid in the patch sequence: raster order, or with windowed attention window by
    /// window (windows in raster order, units in raster order inside each window). Returns the raster index of each sequence slot, with
    /// windows also each slot's window as (first slot, slot count) and, per raster index, its slot; both empty without windows.
    /// </summary>
    private (int[] Units, int[] Spans, int[] Order) MergeUnitOrder(int mh, int mw)
    {
        var units = new int[mh * mw];
        int ws = p.WindowSize;
        if (ws <= 0)
        {
            for (int i = 0; i < units.Length; i++)
            {
                units[i] = i;
            }

            return (units, [], []);
        }

        var spans = new int[units.Length * 2];
        var order = new int[units.Length];
        int slot = 0;
        for (int wy = 0; wy < mh; wy += ws)
        {
            for (int wx = 0; wx < mw; wx += ws)
            {
                int first = slot;
                int count = Math.Min(ws, mh - wy) * Math.Min(ws, mw - wx);
                for (int y = wy; y < Math.Min(wy + ws, mh); y++)
                {
                    for (int x = wx; x < Math.Min(wx + ws, mw); x++, slot++)
                    {
                        units[slot] = y * mw + x;
                        order[y * mw + x] = slot;
                        spans[slot * 2] = first;
                        spans[slot * 2 + 1] = count;
                    }
                }
            }
        }

        return (units, spans, order);
    }

    /// <summary>Bilinear taps (align_corners) resampling the learned g×g position grid to the image grid, per patch.</summary>
    private (int[] Index, float[] Weight) PositionTaps(int[] positions, int gh, int gw)
    {
        int n = positions.Length / 2;
        int side = Math.Max(1, p.PositionGridSide);
        var index = new int[n * 4];
        var weight = new float[n * 4];
        if (p.PositionEmbeddingCount == 0)
        {
            return (index, weight);
        }

        static void Axis(int pos, int count, int side, out int i0, out int i1, out float w0, out float w1)
        {
            double f = count > 1 ? pos * (side - 1.0) / (count - 1) : 0;
            i0 = Math.Min((int)Math.Floor(f), side - 1);
            i1 = Math.Min(i0 + 1, side - 1);
            w1 = (float)(f - i0);
            w0 = 1f - w1;
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
    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1)
        {
            return ((a + 2) * x - (a + 3)) * x * x + 1;
        }

        if (x < 2)
        {
            return (((x - 5) * x + 8) * x - 4) * a;
        }

        return 0;
    }

    private static (int[] Start, int[] Count, float[] Weights, int Max) Coefficients(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = 2.0 * filterScale;
        int max = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[outSize];
        var count = new int[outSize];
        var w = new float[outSize * max];
        for (int x = 0; x < outSize; x++)
        {
            double center = (x + 0.5) * scale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize);
            double sum = 0;
            var tmp = new double[xmax - xmin];
            for (int i = 0; i < tmp.Length; i++)
            {
                tmp[i] = Cubic((xmin + i - center + 0.5) / filterScale);
                sum += tmp[i];
            }
            start[x] = xmin;
            count[x] = tmp.Length;
            for (int i = 0; i < tmp.Length; i++)
            {
                w[x * max + i] = (float)(tmp[i] / sum);
            }
        }
        return (start, count, w, max);
    }

    /// <summary>Antialiased bicubic resize of interleaved 8-bit RGB (rounded and clamped to 8 bits like the reference).</summary>
    internal static byte[] ResizeBicubic(byte[] src, int sw, int sh, int dw, int dh)
    {
        var (hs, hc, hw, hm) = Coefficients(sw, dw);
        var (vs, vc, vw, vm) = Coefficients(sh, dh);
        // Horizontal pass keeps float precision, vertical pass rounds.
        var tmp = new float[(long)sh * dw * 3];
        Parallel.For(0, sh, y =>
        {
            for (int x = 0; x < dw; x++)
            {
                float r = 0;
                float g = 0;
                float b = 0;
                for (int i = 0; i < hc[x]; i++)
                {
                    float w = hw[x * hm + i];
                    int s = (y * sw + hs[x] + i) * 3;
                    r += src[s] * w;
                    g += src[s + 1] * w;
                    b += src[s + 2] * w;
                }
                int o = (y * dw + x) * 3;
                tmp[o] = r;
                tmp[o + 1] = g;
                tmp[o + 2] = b;
            }
        });
        var dst = new byte[(long)dw * dh * 3];
        Parallel.For(0, dh, y =>
        {
            for (int x = 0; x < dw; x++)
            {
                float r = 0;
                float g = 0;
                float b = 0;
                for (int i = 0; i < vc[y]; i++)
                {
                    float w = vw[y * vm + i];
                    int s = ((vs[y] + i) * dw + x) * 3;
                    r += tmp[s] * w;
                    g += tmp[s + 1] * w;
                    b += tmp[s + 2] * w;
                }
                int o = (y * dw + x) * 3;
                dst[o] = Clamp8(r);
                dst[o + 1] = Clamp8(g);
                dst[o + 2] = Clamp8(b);
            }
        });
        return dst;
    }

    private static byte Clamp8(float v) => (byte)Math.Clamp((int)MathF.Round(v, MidpointRounding.AwayFromZero), 0, 255);
}
