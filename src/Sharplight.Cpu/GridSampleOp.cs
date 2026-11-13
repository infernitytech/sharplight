using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>GridSample (ONNX/ORT semantics): linear, nearest, cubic; zeros / border / reflection padding.</summary>
internal sealed unsafe class GridSampleOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _g;
    private readonly TRef _y;
    private readonly int _mode;
    private readonly int _pad;
    private readonly bool _align; // mode 0 linear 1 nearest 2 cubic; pad 0 zeros 1 border 2 reflection
    private readonly int _c;
    private readonly int _h;
    private readonly int _w;
    private readonly int _ho;
    private readonly int _wo;
    private readonly Action<int> _body;
    private long _n;
    private long _rows;
    private long _per;
    public override string Name => "GridSample";

    public GridSampleOp(TRef x, TRef grid, TRef y, string mode, string padding, bool alignCorners)
    {
        _x = x;
        _g = grid;
        _y = y;
        _align = alignCorners;
        _mode = mode switch { "nearest" => 1, "cubic" => 2, _ => 0 };
        _pad = padding switch { "border" => 1, "reflection" => 2, _ => 0 };
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _rows = _n * _ho;
        if (_rows * _wo * _c == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_rows, (long)_wo * _c * 8);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private float Denorm(float v, int len) => _align ? (v + 1f) * (len - 1) / 2f : ((v + 1f) * len - 1f) / 2f;

    private static float Reflect(float x, float lo, float hi)
    {
        float range = hi - lo;
        if (range == 0)
        {
            return lo;
        }

        if (x < lo)
        {
            float dx = lo - x;
            int n = (int)(dx / range);
            float r = dx - n * range;
            return n % 2 == 0 ? lo + r : hi - r;
        }
        if (x > hi)
        {
            float dx = x - hi;
            int n = (int)(dx / range);
            float r = dx - n * range;
            return n % 2 == 0 ? hi - r : lo + r;
        }
        return x;
    }

    /// <summary>Applies the padding mode to a (denormalized) coordinate.</summary>
    private float Coord(float x, int len)
    {
        if (_pad == 1)
        {
            return Math.Clamp(x, 0f, len - 1f);
        }

        if (_pad == 2)
        {
            x = _align ? Reflect(x, 0f, len - 1f) : Reflect(x, -0.5f, len - 0.5f);
            return Math.Clamp(x, 0f, len - 1f);
        }
        return x;
    }

    private float Px(float* img, int r, int c) => (uint)r < (uint)_h && (uint)c < (uint)_w ? img[r * _w + c] : 0f;

    /// <summary>Cubic taps apply the padding mode per sample (ORT PixelAtGrid), not to the coordinate.</summary>
    private float PxPad(float* img, int r, int c)
    {
        if (_pad == 0)
        {
            return Px(img, r, c);
        }

        if (_pad == 1)
        {
            return img[Math.Clamp(r, 0, _h - 1) * _w + Math.Clamp(c, 0, _w - 1)];
        }

        float xmin = _align ? 0f : -0.5f;
        float ymin = xmin;
        float xmax = _align ? _w - 1f : _w - 0.5f;
        float ymax = _align ? _h - 1f : _h - 0.5f;
        int rc = (int)Reflect(r, ymin, ymax);
        int cc = (int)Reflect(c, xmin, xmax);
        return img[rc * _w + cc];
    }

    private static void CubicCoeffs(float t, float* c)
    {
        const float A = -0.75f;
        float CC1(float x) => ((A + 2f) * x - (A + 3f)) * x * x + 1f;
        float CC2(float x) => ((A * x - 5f * A) * x + 8f * A) * x - 4f * A;
        c[0] = CC2(t + 1f);
        c[1] = CC1(t);
        c[2] = CC1(1f - t);
        c[3] = CC2(2f - t);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_rows, s + _per);
        float* coef = stackalloc float[8];
        for (long row = s; row < e; row++)
        {
            long n = row / _ho;
            int oh = (int)(row % _ho);
            for (int ow = 0; ow < _wo; ow++)
            {
                float* gp = (float*)_g.P + ((n * _ho + oh) * _wo + ow) * 2;
                float fx = Denorm(gp[0], _w);
                float fy = Denorm(gp[1], _h);
                if (_mode != 2)
                {
                    fx = Coord(fx, _w);
                    fy = Coord(fy, _h);
                }
                for (int c = 0; c < _c; c++)
                {
                    float* img = (float*)_x.P + (n * _c + c) * _h * _w;
                    float v;
                    if (_mode == 1)
                    {
                        v = Px(img, (int)MathF.Round(fy, MidpointRounding.ToEven), (int)MathF.Round(fx, MidpointRounding.ToEven));
                    }
                    else if (_mode == 0)
                    {
                        float x1 = MathF.Floor(fx);
                        float y1 = MathF.Floor(fy);
                        int xi = (int)x1;
                        int yi = (int)y1;
                        float dx2 = x1 + 1 - fx;
                        float dx1 = fx - x1;
                        float dy2 = y1 + 1 - fy;
                        float dy1 = fy - y1;
                        v = dy2 * (dx2 * Px(img, yi, xi) + dx1 * Px(img, yi, xi + 1)) + dy1 * (dx2 * Px(img, yi + 1, xi) + dx1 * Px(img, yi + 1, xi + 1));
                    }
                    else
                    {
                        float x0 = MathF.Floor(fx) - 1;
                        float y0 = MathF.Floor(fy) - 1;
                        CubicCoeffs(fx - x0 - 1, coef);
                        CubicCoeffs(fy - y0 - 1, coef + 4);
                        int xi = (int)x0;
                        int yi = (int)y0;
                        float acc = 0;
                        for (int h = 0; h < 4; h++)
                        {
                            float r = 0;
                            for (int w = 0; w < 4; w++)
                            {
                                r += coef[w] * PxPad(img, yi + h, xi + w);
                            }

                            acc += coef[4 + h] * r;
                        }
                        v = acc;
                    }
                    ((float*)_y.P)[((n * _c + c) * _ho + oh) * _wo + ow] = v;
                }
            }
        }
    }
}
