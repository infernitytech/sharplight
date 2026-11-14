using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// GridSample (ONNX/ORT semantics): 2-D linear, nearest, cubic and volumetric (5-D) trilinear, nearest; zeros / border / reflection
/// padding. A 2-D sample is a volumetric one with a single depth slice, so both share the row decomposition over (n, do, ho).
/// </summary>
internal sealed unsafe class GridSampleOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _g;
    private readonly TRef _y;
    private readonly int _mode;
    private readonly int _pad;
    private readonly bool _align; // mode 0 linear 1 nearest 2 cubic; pad 0 zeros 1 border 2 reflection
    private readonly int _c;
    private readonly bool _volume;
    private readonly int _d;
    private readonly int _do;
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
        _volume = x.Dims.Length == 5;
        int s = _volume ? 1 : 0;
        _d = _volume ? (int)x.Dims[2] : 1;
        _do = _volume ? (int)y.Dims[2] : 1;
        _h = (int)x.Dims[2 + s];
        _w = (int)x.Dims[3 + s];
        _ho = (int)y.Dims[2 + s];
        _wo = (int)y.Dims[3 + s];
        _body = _volume ? VolumeBody : Body;
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _rows = _n * _do * _ho;
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

    private float Vx(float* vol, int z, int r, int c) =>
        (uint)z < (uint)_d && (uint)r < (uint)_h && (uint)c < (uint)_w ? vol[((long)z * _h + r) * _w + c] : 0f;

    private void VolumeBody(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_rows, s + _per);
        long plane = (long)_d * _h * _w;
        long outPlane = (long)_do * _ho * _wo;
        for (long row = s; row < e; row++)
        {
            long n = row / ((long)_do * _ho);
            long inner = row % ((long)_do * _ho);
            for (int ow = 0; ow < _wo; ow++)
            {
                long point = inner * _wo + ow;
                float* gp = (float*)_g.P + (n * outPlane + point) * 3;
                float fx = Coord(Denorm(gp[0], _w), _w);
                float fy = Coord(Denorm(gp[1], _h), _h);
                float fz = Coord(Denorm(gp[2], _d), _d);
                for (int c = 0; c < _c; c++)
                {
                    float* vol = (float*)_x.P + (n * _c + c) * plane;
                    float v;
                    if (_mode == 1)
                    {
                        v = Vx(vol, (int)MathF.Round(fz, MidpointRounding.ToEven), (int)MathF.Round(fy, MidpointRounding.ToEven),
                            (int)MathF.Round(fx, MidpointRounding.ToEven));
                    }
                    else
                    {
                        float x1 = MathF.Floor(fx);
                        float y1 = MathF.Floor(fy);
                        float z1 = MathF.Floor(fz);
                        int xi = (int)x1;
                        int yi = (int)y1;
                        int zi = (int)z1;
                        float dx = fx - x1;
                        float dy = fy - y1;
                        float dz = fz - z1;
                        v = 0;
                        for (int k = 0; k < 8; k++)
                        {
                            int bx = k & 1;
                            int by = (k >> 1) & 1;
                            int bz = k >> 2;
                            float weight = (bx == 1 ? dx : 1 - dx) * (by == 1 ? dy : 1 - dy) * (bz == 1 ? dz : 1 - dz);
                            v += weight * Vx(vol, zi + bz, yi + by, xi + bx);
                        }
                    }

                    ((float*)_y.P)[(n * _c + c) * outPlane + point] = v;
                }
            }
        }
    }
}
