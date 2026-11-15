using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D resize (nearest / bilinear) with ONNX Runtime coordinate semantics (single precision), over dense NCHW planes, or over
/// whole pixels (channel vectors) when the input is a channels-last view and the output channels-last
/// (<see cref="ChannelsLast"/>).
/// </summary>
internal sealed unsafe class ResizeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _linear;
    private readonly string _coord;
    private readonly string _nearest;
    private readonly int _h;
    private readonly int _w;
    private readonly int _oh;
    private readonly int _ow;
    private readonly int[] _yi0;
    private readonly int[] _yi1;
    private readonly int[] _xi0;
    private readonly int[] _xi1;
    private readonly float[] _yw;
    private readonly float[] _xw;
    private readonly Action<int> _body;
    private readonly bool _channelsLast;
    private readonly int _c;
    private readonly long _ps;
    private readonly long _rs;
    private readonly long _is;
    private long _planes;
    private long _per;
    public override string Name => "Resize." + (_linear ? "linear" : "nearest");

    public ResizeOp(TRef x, TRef y, string mode, string coord, string nearestMode)
    {
        _x = x;
        _y = y;
        _linear = mode == "linear";
        _coord = coord;
        _nearest = nearestMode;
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _oh = (int)y.Dims[2];
        _ow = (int)y.Dims[3];
        (_yi0, _yi1, _yw) = Axis(_h, _oh);
        (_xi0, _xi1, _xw) = Axis(_w, _ow);
        _channelsLast = ChannelsLast.IsView(x) && ChannelsLast.Is(y);
        _c = (int)x.Dims[1];
        _ps = _channelsLast ? ChannelsLast.PixelStride(x) : 0;
        _rs = _channelsLast ? ChannelsLast.RowStride(x) : 0;
        _is = _channelsLast ? ChannelsLast.ImageStride(x) : 0;
        _body = _channelsLast ? RowBody : Body;
    }

    private float Original(int xr, int lenIn, int lenOut)
    {
        float scale = (float)lenOut / lenIn;
        switch (_coord)
        {
            case "align_corners":
                return lenOut == 1 ? 0f : (float)xr * (lenIn - 1) / (lenOut - 1);
            case "asymmetric":
                return xr / scale;
            case "pytorch_half_pixel":
                return lenOut > 1 ? (xr + 0.5f) / scale - 0.5f : 0f;
            default:
                return (xr + 0.5f) / scale - 0.5f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private (int[], int[], float[]) Axis(int lenIn, int lenOut)
    {
        var i0 = new int[lenOut];
        var i1 = new int[lenOut];
        var w = new float[lenOut];
        for (int o = 0; o < lenOut; o++)
        {
            float x = Original(o, lenIn, lenOut);
            if (_linear)
            {
                x = Math.Max(0f, Math.Min(x, lenIn - 1));
                int a = Math.Min((int)x, lenIn - 1);
                int b = Math.Min(a + 1, lenIn - 1);
                i0[o] = a;
                i1[o] = b;
                w[o] = x - a;
            }
            else
            {
                long idx = _nearest switch
                {
                    "floor" => (long)MathF.Floor(x),
                    "ceil" => (long)MathF.Ceiling(x),
                    "round_prefer_ceil" => x == (long)x + 0.5f ? (long)MathF.Ceiling(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
                    _ => x == (long)x + 0.5f ? (long)MathF.Floor(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
                };
                i0[o] = (int)Math.Clamp(idx, 0, lenIn - 1);
            }
        }
        return (i0, i1, w);
    }

    public override void Run(RunParameters p)
    {
        _planes = _x.Dim(0, p) * _x.Dims[1];
        if (_planes == 0)
        {
            return;
        }

        if (_channelsLast)
        {
            CpuParallel.For((int)(_x.Dim(0, p) * _oh), _body);
            return;
        }

        int chunks = (int)Math.Min(_planes, CpuParallel.Chunks(_planes * _oh * _ow, 16384));
        _per = (_planes + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_planes, s + _per);
        for (long pl = s; pl < e; pl++)
        {
            float* xp = (float*)_x.P + pl * _h * _w;
            float* yp = (float*)_y.P + pl * _oh * _ow;
            for (int oy = 0; oy < _oh; oy++)
            {
                float* y = yp + (long)oy * _ow;
                if (!_linear)
                {
                    float* r = xp + (long)_yi0[oy] * _w;
                    for (int ox = 0; ox < _ow; ox++)
                    {
                        y[ox] = r[_xi0[ox]];
                    }
                }
                else
                {
                    float* r0 = xp + (long)_yi0[oy] * _w;
                    float* r1 = xp + (long)_yi1[oy] * _w;
                    float dy = _yw[oy];
                    float dy2 = 1f - dy;
                    for (int ox = 0; ox < _ow; ox++)
                    {
                        float dx = _xw[ox];
                        float dx2 = 1f - dx;
                        int a = _xi0[ox];
                        int b = _xi1[ox];
                        y[ox] = r0[a] * dx2 * dy2 + r0[b] * dx * dy2 + r1[a] * dx2 * dy + r1[b] * dx * dy;
                    }
                }
            }
        }
    }

    /// <summary>Channels-last: output row <paramref name="task"/> (image * OH + row), one channel vector of a pixel at a time.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RowBody(int task)
    {
        long n = task / _oh;
        int oy = (int)(task - n * _oh);
        float* xn = (float*)_x.P + n * _is;
        float* y = (float*)_y.P + ((n * _oh + oy) * _ow) * _c;
        int c = _c;
        if (!_linear)
        {
            float* r = xn + _yi0[oy] * _rs;
            for (int ox = 0; ox < _ow; ox++)
            {
                new ReadOnlySpan<float>(r + _xi0[ox] * _ps, c).CopyTo(new Span<float>(y + (long)ox * c, c));
            }

            return;
        }

        float* r0 = xn + _yi0[oy] * _rs;
        float* r1 = xn + _yi1[oy] * _rs;
        float dy = _yw[oy];
        float dy2 = 1f - dy;
        for (int ox = 0; ox < _ow; ox++)
        {
            float dx = _xw[ox];
            float dx2 = 1f - dx;
            float* a0 = r0 + _xi0[ox] * _ps;
            float* b0 = r0 + _xi1[ox] * _ps;
            float* a1 = r1 + _xi0[ox] * _ps;
            float* b1 = r1 + _xi1[ox] * _ps;
            float* d = y + (long)ox * c;
            int k = 0;
            if (CpuIsa.Use512)
            {
                // The scalar expression, lane-wise (no contraction), so both layouts give identical results.
                var w00 = Vector512.Create(dx2);
                var w01 = Vector512.Create(dx);
                var v0 = Vector512.Create(dy2);
                var v1 = Vector512.Create(dy);
                for (; k + 16 <= c; k += 16)
                {
                    var t = Vector512.Load(a0 + k) * w00 * v0 + Vector512.Load(b0 + k) * w01 * v0 + Vector512.Load(a1 + k) * w00 * v1
                        + Vector512.Load(b1 + k) * w01 * v1;
                    t.Store(d + k);
                }
            }

            for (; k < c; k++)
            {
                d[k] = a0[k] * dx2 * dy2 + b0[k] * dx * dy2 + a1[k] * dx2 * dy + b1[k] * dx * dy;
            }
        }
    }
}
