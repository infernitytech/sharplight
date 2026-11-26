namespace Sharplight.Vision;

/// <summary>
/// Turns raw RGB pixels into the inputs of a model's vision encoder, driven only by the canonical
/// <see cref="VisionParameters"/>: aspect-preserving "smart" resize to a multiple of patch_size·merge_size within the
/// pixel bounds, or to the fixed square <c>image_size</c> (antialiased bicubic), per-channel normalization, patchification (the still image is repeated to fill
/// temporal_patch_size frames) in spatial-merge-window order (window by window when the tower has windowed attention), plus the position inputs (row/col per patch and the
/// bilinear taps into the learned position grid, align_corners semantics) and, for towers with row separators or boundary rows, the output
/// layout. Two-stage towers of fixed-size views (<see cref="VisionParameters.Sam"/>) get a global view and local tiles instead (see
/// <see cref="PreprocessedImage.TileColumns"/>).
/// </summary>
public sealed class VisionPreprocessor(VisionParameters p, ImageTokenLimits? limits = null)
{
    /// <summary>The vision tower parameters this preprocessor follows.</summary>
    public VisionParameters Parameters => p;

    /// <summary>Target size in pixels for an input size (smart resize).</summary>
    internal (int Width, int Height) TargetSize(int width, int height)
    {
        if (p.ImageSize > 0)
        {
            // A fixed square (CogVLM), whatever the aspect ratio.
            return (p.ImageSize, p.ImageSize);
        }

        int factor = p.PatchSize * p.SpatialMergeSize;
        long unit = (long)factor * factor;
        long minPixels = limits?.MinTokens is int mt ? mt * unit : p.MinPixels;
        long maxPixels = limits?.MaxTokens is int xt ? xt * unit : p.MaxPixels;
        double h = height;
        double w = width;
        if (p.ResizeMode == VisionParameters.FitBudgetResize)
        {
            // Gemma 4: the largest image within the pixel budget (also enlarging a small one), both sides rounded down to a multiple of the factor.
            double scale = Math.Sqrt((double)maxPixels / (h * w));
            int fh = (int)Math.Floor(scale * h / factor) * factor;
            int fw = (int)Math.Floor(scale * w / factor) * factor;
            int maxSide = (int)(maxPixels / unit) * factor;
            if (fh == 0 && fw == 0)
            {
                throw new ArgumentException($"Image {width}x{height} is too small for the pixel budget {maxPixels}.");
            }

            if (fh == 0)
            {
                fh = factor;
                fw = Math.Min((int)Math.Floor(w / h) * factor, maxSide);
            }
            else if (fw == 0)
            {
                fw = factor;
                fh = Math.Min((int)Math.Floor(h / w) * factor, maxSide);
            }

            return (fw, fh);
        }

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

        if (p.Sam is not null)
        {
            return ViewPreprocessor.Process(p, image);
        }

        if (p.TileSize > 0)
        {
            return ProcessTiles(image);
        }

        var (tw, th) = TargetSize(image.Width, image.Height);
        byte[] px = image.Width == tw && image.Height == th ? image.Rgb : Resize(image.Rgb, image.Width, image.Height, tw, th, p.ResizeFilter);
        return Patchify(Normalize(px), tw, th);
    }

    // Normalization: (x/255 - mean) / std  ==  x * a + b.
    private float[] Normalize(byte[] px)
    {
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

        return values;
    }

    /// <summary>
    /// Size of a tiled tower's image before it is cut (Idefics3): the longer side resized to <see cref="VisionParameters.LongestEdge"/> (the shorter
    /// one truncated, then rounded up to even, at most 4096), then both sides rounded up to multiples of the tile size, the shorter one recomputed from
    /// the aspect ratio first.
    /// </summary>
    internal (int Width, int Height, int ResizedWidth, int ResizedHeight) TiledSize(int width, int height)
    {
        int L = p.LongestEdge;
        int T = p.TileSize;
        int w;
        int h;
        if (width >= height)
        {
            w = L;
            h = (int)(w / ((double)width / height));
            h += h % 2;
        }
        else
        {
            h = L;
            w = (int)(h * ((double)width / height));
            w += w % 2;
        }

        w = Math.Clamp(w, 1, 4096);
        h = Math.Clamp(h, 1, 4096);
        double aspect = (double)w / h;
        int rw;
        int rh;
        if (w >= h)
        {
            rw = (w + T - 1) / T * T;
            rh = ((int)(rw / aspect) + T - 1) / T * T;
        }
        else
        {
            rh = (h + T - 1) / T * T;
            rw = ((int)(rh * aspect) + T - 1) / T * T;
        }

        return (rw, rh, w, h);
    }

    /// <summary>Tiled towers (Idefics3): resizes the image, cuts it into tiles and adds the overview (see <see cref="VisionParameters.TileSize"/>).</summary>
    private PreprocessedImage ProcessTiles(ImageInput image)
    {
        int T = p.TileSize;
        var (rw, rh, w, h) = TiledSize(image.Width, image.Height);
        var resized = image.Width == w && image.Height == h ? image.Rgb : Resize(image.Rgb, image.Width, image.Height, w, h, p.ResizeFilter);
        var refined = w == rw && h == rh ? resized : Resize(resized, w, h, rw, rh, p.ResizeFilter);
        int columns = rw / T;
        int rows = rh / T;
        if (columns == 1 && rows == 1)
        {
            return PatchifyTiles(Normalize(refined), [], 0, 0);
        }

        var tiles = new List<float[]>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                var tile = new byte[3 * T * T];
                for (int y = 0; y < T; y++)
                {
                    Array.Copy(refined, 3 * ((r * T + y) * rw + c * T), tile, 3 * y * T, 3 * T);
                }

                tiles.Add(Normalize(tile));
            }
        }

        var overview = Resize(refined, rw, rh, T, T, p.ResizeFilter);
        return PatchifyTiles(Normalize(overview), tiles, columns, rows);
    }

    /// <summary>
    /// The encoder input of a tiled tower from normalized interleaved RGB views of tile size: the tiles of a <paramref name="columns"/> ×
    /// <paramref name="rows"/> grid in raster order (none: 0 × 0), then the overview. Each view is patchified on its own and is one attention
    /// window; <see cref="PreprocessedImage.GridHeight"/> stacks the views' patch grids and <see cref="PreprocessedImage.MergedHeight"/> their
    /// merged grids, so the output rows are the views' in order.
    /// </summary>
    internal PreprocessedImage PatchifyTiles(float[] overview, IReadOnlyList<float[]> tiles, int columns, int rows)
    {
        int T = p.TileSize;
        if (tiles.Count != columns * rows || overview.Length != 3 * T * T || tiles.Any(t => t.Length != 3 * T * T))
        {
            throw new ArgumentException($"Tiled image: expected {columns}x{rows} tiles and an overview of {T}x{T} pixels.");
        }

        var views = tiles.Select(t => Patchify(t, T, T)).Append(Patchify(overview, T, T)).ToArray();
        int n = views[0].PatchCount;
        var spans = new int[views.Length * n * 2];
        for (int v = 0; v < views.Length; v++)
        {
            for (int i = 0; i < n; i++)
            {
                spans[(v * n + i) * 2] = v * n;
                spans[(v * n + i) * 2 + 1] = n;
            }
        }

        var first = views[0];
        return new PreprocessedImage
        {
            Width = columns > 0 ? columns * T : T, Height = rows > 0 ? rows * T : T,
            GridHeight = first.GridHeight * views.Length, GridWidth = first.GridWidth,
            MergedHeight = first.MergedHeight * views.Length, MergedWidth = first.MergedWidth,
            Patches = [.. views.SelectMany(v => v.Patches)], Positions = [.. views.SelectMany(v => v.Positions)],
            EmbeddingIndex = [.. views.SelectMany(v => v.EmbeddingIndex)], EmbeddingWeight = [.. views.SelectMany(v => v.EmbeddingWeight)],
            WindowSpans = spans, TileColumns = columns, TileRows = rows,
        };
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
            Layout = OutputLayout(gh / m, gw / m),
        };
    }

    /// <summary>
    /// The output rows of a tower with row separators or boundary rows (else empty), as rows of the table
    /// [merged rows | class row | separator | begin | end]: begin, each row of merged tokens followed by the separator, end (the class row is not output).
    /// </summary>
    private int[] OutputLayout(int mh, int mw)
    {
        if (!p.HasLayout)
        {
            return [];
        }

        int merged = mh * mw;
        int separator = merged + (p.ClassToken ? 1 : 0);
        int begin = separator + (p.RowSeparator ? 1 : 0);
        var layout = new List<int>(p.OutputTokenCount(mh, mw));
        if (p.ImageBoundary)
        {
            layout.Add(begin);
        }

        for (int r = 0; r < mh; r++)
        {
            for (int c = 0; c < mw; c++)
            {
                layout.Add(r * mw + c);
            }

            if (p.RowSeparator)
            {
                layout.Add(separator);
            }
        }

        if (p.ImageBoundary)
        {
            layout.Add(begin + 1);
        }

        return [.. layout];
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

    /// <summary>Taps resampling the learned g×g position grid to the image grid, per patch (<see cref="VisionParameters.PositionInterpolation"/>).</summary>
    private (int[] Index, float[] Weight) PositionTaps(int[] positions, int gh, int gw)
    {
        int n = positions.Length / 2;
        int side = Math.Max(1, p.PositionGridSide);
        int taps = p.PositionTaps;
        var index = new int[n * taps];
        var weight = new float[n * taps];
        if (p.PositionEmbeddingCount == 0)
        {
            return (index, weight);
        }

        if (p.AxialPositionEmbedding)
        {
            // Two tables of PositionEmbeddingCount rows, x (the column) first, then y (the row); the embeddings are summed.
            for (int i = 0; i < n; i++)
            {
                index[i * 2] = positions[i * 2 + 1];
                index[i * 2 + 1] = p.PositionEmbeddingCount + positions[i * 2];
                weight[i * 2] = 1f;
                weight[i * 2 + 1] = 1f;
            }

            return (index, weight);
        }

        if (taps == 16)
        {
            BicubicTaps(positions, gh, gw, side, index, weight);
            return (index, weight);
        }

        bool halfPixel = p.PositionInterpolation == "bilinear_half_pixel";
        // align_corners: the grid's corners on the image's corner patches; half pixel (PyTorch's resize, align_corners off): source coordinate
        // (pos + 0.5)·side/count − 0.5, clamped at 0.
        static void Axis(int pos, int count, int side, bool halfPixel, out int i0, out int i1, out float w0, out float w1)
        {
            double f = halfPixel ? Math.Max(0.0, (pos + 0.5) * side / count - 0.5) : count > 1 ? pos * (side - 1.0) / (count - 1) : 0;
            i0 = Math.Min((int)Math.Floor(f), side - 1);
            i1 = Math.Min(i0 + 1, side - 1);
            w1 = (float)(f - i0);
            w0 = 1f - w1;
        }
        for (int i = 0; i < n; i++)
        {
            Axis(positions[i * 2], gh, side, halfPixel, out int r0, out int r1, out float wr0, out float wr1);
            Axis(positions[i * 2 + 1], gw, side, halfPixel, out int c0, out int c1, out float wc0, out float wc1);
            index[i * 4 + 0] = r0 * side + c0; weight[i * 4 + 0] = wr0 * wc0;
            index[i * 4 + 1] = r0 * side + c1; weight[i * 4 + 1] = wr0 * wc1;
            index[i * 4 + 2] = r1 * side + c0; weight[i * 4 + 2] = wr1 * wc0;
            index[i * 4 + 3] = r1 * side + c1; weight[i * 4 + 3] = wr1 * wc1;
        }
        return (index, weight);
    }

    /// <summary>
    /// PyTorch's bicubic grid_sample of the patch centres (align_corners off, border padding, A = −0.75): source coordinate
    /// (pos + 0.5)·side/count − 0.5, 4 taps per axis from floor − 1 to floor + 2, each tap clamped to the grid.
    /// </summary>
    private static void BicubicTaps(int[] positions, int gh, int gw, int side, int[] index, float[] weight)
    {
        static double Cubic1(double x, double a) => ((a + 2) * x - (a + 3)) * x * x + 1;          // |x| <= 1
        static double Cubic2(double x, double a) => ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;    // 1 < |x| < 2
        static void Axis(int pos, int count, int side, Span<int> idx, Span<double> w)
        {
            const double A = -0.75;
            double f = (pos + 0.5) * side / count - 0.5;
            int f0 = (int)Math.Floor(f);
            double t = f - f0;
            w[0] = Cubic2(t + 1, A);
            w[1] = Cubic1(t, A);
            w[2] = Cubic1(1 - t, A);
            w[3] = Cubic2(2 - t, A);
            for (int k = 0; k < 4; k++)
            {
                idx[k] = Math.Clamp(f0 - 1 + k, 0, side - 1);
            }
        }

        Span<int> ri = stackalloc int[4];
        Span<int> ci = stackalloc int[4];
        Span<double> rw = stackalloc double[4];
        Span<double> cw = stackalloc double[4];
        int n = positions.Length / 2;
        for (int i = 0; i < n; i++)
        {
            Axis(positions[i * 2], gh, side, ri, rw);
            Axis(positions[i * 2 + 1], gw, side, ci, cw);
            for (int y = 0; y < 4; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    index[i * 16 + y * 4 + x] = ri[y] * side + ci[x];
                    weight[i * 16 + y * 4 + x] = (float)(rw[y] * cw[x]);
                }
            }
        }
    }

    // ---- antialiased bicubic resize (PIL / torch antialias=True semantics, Keys kernel a = -0.5; see ImageResampling) ----------
    /// <summary>Antialiased bicubic resize of interleaved 8-bit RGB (rounded and clamped to 8 bits like the reference).</summary>
    internal static byte[] ResizeBicubic(byte[] src, int sw, int sh, int dw, int dh) => Resize(src, sw, sh, dw, dh, ImageResampleFilter.Bicubic);

    /// <summary>Antialiased resize of interleaved 8-bit RGB with PIL's bilinear or bicubic filter (rounded and clamped to 8 bits after each pass).</summary>
    internal static byte[] Resize(byte[] src, int sw, int sh, int dw, int dh, ImageResampleFilter filter)
    {
        var (hs, hc, hw, hm) = ImageResampling.Coefficients(sw, dw, filter);
        var (vs, vc, vw, vm) = ImageResampling.Coefficients(sh, dh, filter);
        // Both passes round and clamp to 8 bits, as PIL (and torchvision's antialiased resize of uint8 images) does; the intermediate clamp
        // matters next to sharp edges, where the kernel overshoots.
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
                tmp[o] = Clamp8(r);
                tmp[o + 1] = Clamp8(g);
                tmp[o + 2] = Clamp8(b);
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
