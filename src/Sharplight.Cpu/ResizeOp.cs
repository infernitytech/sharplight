using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>2-D resize (nearest / bilinear) with ONNX Runtime coordinate semantics (single precision).</summary>
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
        _body = Body;
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

        int chunks = (int)Math.Min(_planes, CpuParallel.Chunks(_planes * _oh * _ow, 16384));
        _per = (_planes + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

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
}
