namespace Sharplight.Cpu.Tests;

/// <summary>Scalar (double precision) references for the spatial / reduction / batched-matmul ops (NCHW layout, ONNX semantics).</summary>
internal static class SRef
{
    public static long Prod(IEnumerable<long> d) => d.Aggregate(1L, (a, b) => a * b);

    public static long OutDim(long n, long k, long s, long p0, long p1, long d, bool ceil)
    {
        long span = n + p0 + p1 - d * (k - 1) - 1;
        long o = (ceil ? (span + s - 1) / s : span / s) + 1;
        if (ceil && (o - 1) * s >= n + p0)
        {
            o--;
        }

        return o;
    }

    public static float[] Conv(float[] x, int N, int C, int H, int W, float[] w, int O, int KH, int KW, float[]? bias,
        int[] stride, int[] pad, int[] dil, int groups, out int HO, out int WO)
    {
        HO = (int)OutDim(H, KH, stride[0], pad[0], pad[2], dil[0], false); WO = (int)OutDim(W, KW, stride[1], pad[1], pad[3], dil[1], false);
        int Cg = C / groups, Og = O / groups;
        var y = new float[N * O * HO * WO];
        for (int n = 0; n < N; n++)
        {
            for (int o = 0; o < O; o++)
            {
                int g = o / Og;
                for (int oh = 0; oh < HO; oh++)
                {
                    for (int ow = 0; ow < WO; ow++)
                    {
                        double s = bias?[o] ?? 0;
                        for (int c = 0; c < Cg; c++)
                        {
                            for (int kh = 0; kh < KH; kh++)
                            {
                                for (int kw = 0; kw < KW; kw++)
                                {
                                    int ih = oh * stride[0] - pad[0] + kh * dil[0], iw = ow * stride[1] - pad[1] + kw * dil[1];
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    {
                                        continue;
                                    }

                                    s += (double)x[((n * C + g * Cg + c) * H + ih) * W + iw] * w[((o * Cg + c) * KH + kh) * KW + kw];
                                }
                            }
                        }

                        y[((n * O + o) * HO + oh) * WO + ow] = (float)s;
                    }
                }
            }
        }

        return y;
    }

    public static float[] Pool(float[] x, int N, int C, int H, int W, bool max, int[] k, int[] stride, int[] pad, int[] dil, bool ceil, bool includePad, out int HO, out int WO)
    {
        HO = (int)OutDim(H, k[0], stride[0], pad[0], pad[2], dil[0], ceil); WO = (int)OutDim(W, k[1], stride[1], pad[1], pad[3], dil[1], ceil);
        var y = new float[N * C * HO * WO];
        for (int nc = 0; nc < N * C; nc++)
        {
            for (int oh = 0; oh < HO; oh++)
            {
                for (int ow = 0; ow < WO; ow++)
                {
                    int hs = oh * stride[0] - pad[0], ws = ow * stride[1] - pad[1];
                    double best = double.NegativeInfinity, sum = 0; int cnt = 0;
                    for (int kh = 0; kh < k[0]; kh++)
                    {
                        for (int kw = 0; kw < k[1]; kw++)
                        {
                            int ih = hs + kh * dil[0], iw = ws + kw * dil[1];
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                            {
                                continue;
                            }

                            double v = x[(nc * H + ih) * W + iw];
                            best = Math.Max(best, v); sum += v; cnt++;
                        }
                    }

                    double r;
                    if (max)
                    {
                        r = best;
                    }
                    else if (includePad)
                    {
                        // padded-window area: clipped to [-pad_begin, size + pad_end)
                        int he = Math.Min(hs + k[0], H + pad[2]), we = Math.Min(ws + k[1], W + pad[3]);
                        r = sum / ((he - hs) * (we - ws));
                    }
                    else
                    {
                        r = sum / cnt;
                    }

                    y[(nc * HO + oh) * WO + ow] = (float)r;
                }
            }
        }

        return y;
    }

    public static float[] Reduce(float[] x, long[] dims, long[] axes, string mode, bool keep, out long[] od)
    {
        int r = dims.Length;
        var red = new bool[r]; foreach (var a in axes)
        {
            red[a < 0 ? a + r : a] = true;
        }

        od = Enumerable.Range(0, r).Where(i => keep || !red[i]).Select(i => red[i] ? 1L : dims[i]).ToArray();
        var keptDims = Enumerable.Range(0, r).Where(i => !red[i]).Select(i => dims[i]).ToArray();
        long nOut = Prod(keptDims);
        var acc = new double[nOut];
        double init = mode == "max" ? double.NegativeInfinity : mode == "min" ? double.PositiveInfinity : 0;
        Array.Fill(acc, init);
        long total = Prod(dims), cnt = total / Math.Max(1, nOut);
        var idx = new long[r];
        for (long lin = 0; lin < total; lin++)
        {
            long rem = lin; for (int d = r - 1; d >= 0; d--) { idx[d] = rem % dims[d]; rem /= dims[d]; }
            long o = 0;
            for (int d = 0; d < r; d++)
            {
                if (!red[d])
                {
                    o = o * dims[d] + idx[d];
                }
            }

            double v = x[lin];
            acc[o] = mode switch { "max" => Math.Max(acc[o], v), "min" => Math.Min(acc[o], v), _ => acc[o] + v };
        }
        return acc.Select(v => (float)(mode == "mean" ? v / cnt : v)).ToArray();
    }

    /// <summary>a [..., M, K] (or [..., K, M]) times b [..., K, N] (or [..., N, K]) with broadcast batch dims.</summary>
    public static float[] Bmm(float[] a, long[] ad, float[] b, long[] bd, bool ta, bool tb, float alpha, out long[] od)
    {
        int ra = ad.Length, rb = bd.Length, br = Math.Max(ra, rb) - 2;
        long M = ta ? ad[ra - 1] : ad[ra - 2], K = ta ? ad[ra - 2] : ad[ra - 1], N = tb ? bd[rb - 2] : bd[rb - 1];
        var batch = new long[br];
        for (int i = 0; i < br; i++)
        {
            int ia = i - (br - (ra - 2)), ib = i - (br - (rb - 2));
            batch[i] = Math.Max(ia >= 0 ? ad[ia] : 1, ib >= 0 ? bd[ib] : 1);
        }
        od = [.. batch, M, N];
        long nb = Prod(batch);
        var y = new float[nb * M * N];
        for (long bi = 0; bi < nb; bi++)
        {
            long rem = bi, oa = 0, ob = 0, sa = 1, sb = 1;
            for (int i = br - 1; i >= 0; i--)
            {
                long ix = rem % batch[i]; rem /= batch[i];
                int ia = i - (br - (ra - 2)), ib = i - (br - (rb - 2));
                if (ia >= 0) { oa += (ad[ia] == 1 ? 0 : ix) * sa; sa *= ad[ia]; }
                if (ib >= 0) { ob += (bd[ib] == 1 ? 0 : ix) * sb; sb *= bd[ib]; }
            }
            long abase = oa * ad[ra - 2] * ad[ra - 1], bbase = ob * bd[rb - 2] * bd[rb - 1];
            for (long m = 0; m < M; m++)
            {
                for (long n = 0; n < N; n++)
                {
                    double s = 0;
                    for (long k = 0; k < K; k++)
                    {
                        double av = ta ? a[abase + k * M + m] : a[abase + m * K + k];
                        double bv = tb ? b[bbase + n * K + k] : b[bbase + k * N + n];
                        s += av * bv;
                    }
                    y[(bi * M + m) * N + n] = (float)(alpha * s);
                }
            }
        }
        return y;
    }

    private static long PadMap(long o, long pb, long n, string mode)
    {
        long i = o - pb;
        if (i >= 0 && i < n)
        {
            return i;
        }

        if (mode == "constant")
        {
            return -1;
        }

        if (mode == "edge")
        {
            return Math.Clamp(i, 0, n - 1);
        }
        // reflect (no edge repeat): bounce between 0 and n-1
        if (n == 1)
        {
            return 0;
        }

        while (i < 0 || i >= n) { if (i < 0)
            {
                i = -i;
            }

            if (i >= n)
            {
                i = 2 * (n - 1) - i;
            }
        }
        return i;
    }

    public static float[] Pad(float[] x, long[] dims, long[] pads, string mode, float value, out long[] od)
    {
        int r = dims.Length;
        od = Enumerable.Range(0, r).Select(i => dims[i] + pads[i] + pads[i + r]).ToArray();
        var y = new float[Prod(od)];
        var idx = new long[r];
        for (long lin = 0; lin < y.Length; lin++)
        {
            long rem = lin; for (int d = r - 1; d >= 0; d--) { idx[d] = rem % od[d]; rem /= od[d]; }
            long so = 0; bool fill = false;
            for (int d = 0; d < r; d++)
            {
                long m = PadMap(idx[d], pads[d], dims[d], mode);
                if (m < 0) { fill = true; break; }
                so = so * dims[d] + m;
            }
            y[lin] = fill ? value : x[so];
        }
        return y;
    }

    // ---- Resize (ONNX Runtime, float arithmetic) -------------------------------------------------
    private static float Coord(string coord, int xr, int lenIn, int lenOut)
    {
        float scale = (float)lenOut / lenIn;
        return coord switch
        {
            "half_pixel" => (xr + 0.5f) / scale - 0.5f,
            "pytorch_half_pixel" => lenOut > 1 ? (xr + 0.5f) / scale - 0.5f : 0f,
            "align_corners" => lenOut == 1 ? 0f : xr * (float)(lenIn - 1) / (lenOut - 1),
            "asymmetric" => xr / scale,
            _ => throw new ArgumentException(coord),
        };
    }

    private static int Nearest(string mode, float x)
    {
        switch (mode)
        {
            case "floor": return (int)Math.Floor(x);
            case "ceil": return (int)Math.Ceiling(x);
            case "round_prefer_ceil":
                return x == (long)x + 0.5f ? (int)Math.Ceiling(x) : (int)MathF.Round(x, MidpointRounding.AwayFromZero);
            default:
                return x == (long)x + 0.5f ? (int)Math.Floor(x) : (int)MathF.Round(x, MidpointRounding.AwayFromZero);
        }
    }

    public static float[] Resize(float[] x, int N, int C, int H, int W, int OH, int OW, string mode, string coord, string nearest)
    {
        var y = new float[N * C * OH * OW];
        for (int nc = 0; nc < N * C; nc++)
        {
            for (int oy = 0; oy < OH; oy++)
            {
                for (int ox = 0; ox < OW; ox++)
                {
                    float fy = Coord(coord, oy, H, OH), fx = Coord(coord, ox, W, OW);
                    float v;
                    if (mode == "nearest")
                    {
                        int iy = Math.Clamp(Nearest(nearest, fy), 0, H - 1), ix = Math.Clamp(Nearest(nearest, fx), 0, W - 1);
                        v = x[(nc * H + iy) * W + ix];
                    }
                    else
                    {
                        fy = Math.Clamp(fy, 0, H - 1); fx = Math.Clamp(fx, 0, W - 1);
                        int y0 = (int)Math.Floor(fy), x0 = (int)Math.Floor(fx);
                        int y1 = Math.Min(y0 + 1, H - 1), x1 = Math.Min(x0 + 1, W - 1);
                        double dy = fy - y0, dx = fx - x0;
                        double Px(int yy, int xx) => x[(nc * H + yy) * W + xx];
                        v = (float)((1 - dy) * ((1 - dx) * Px(y0, x0) + dx * Px(y0, x1)) + dy * ((1 - dx) * Px(y1, x0) + dx * Px(y1, x1)));
                    }
                    y[(nc * OH + oy) * OW + ox] = v;
                }
            }
        }

        return y;
    }
}
