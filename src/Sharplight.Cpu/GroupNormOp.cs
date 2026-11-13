using System.Numerics;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Group normalization over NCHW input with optional per-channel scale and bias.</summary>
internal sealed unsafe class GroupNormOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _w;
    private readonly TRef? _b;
    private readonly int _groups;
    private readonly float _eps;
    private readonly Action<int> _body;
    private long _n;
    private long _c;
    private long _s;
    private long _per;
    public override string Name => "GroupNorm";
    public GroupNormOp(TRef x, TRef? w, TRef? b, TRef y, int groups, float eps)
    {
        _x = x;
        _w = w;
        _b = b;
        _y = y;
        _groups = groups;
        _eps = eps;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _n = d[0];
        _c = d[1];
        _s = OpHelpers.Prod(d, 2, d.Length);
        long items = _n * _groups;
        if (items * _c * _s == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(items, _c / _groups * _s * 4);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s0 = chunk * _per;
        long e0 = Math.Min(_n * _groups, s0 + _per);
        long cg = _c / _groups;
        long cnt = cg * _s;
        float* x = (float*)_x.P;
        float* y = (float*)_y.P;
        float* w = _w is null ? null : (float*)_w.P;
        float* bi = _b is null ? null : (float*)_b.P;
        for (long it = s0; it < e0; it++)
        {
            long n = it / _groups;
            long g = it % _groups;
            long baseOff = (n * _c + g * cg) * _s;
            float* xb = x + baseOff;
            float mean = Simd.Sum(xb, cnt) / cnt;
            double v = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var mv = Vector256.Create(mean);
                var acc = Vector256<float>.Zero;
                long i = 0;
                for (; i + 8 <= cnt; i += 8)
                {
                    var dv = Vector256.Load(xb + i) - mv;
                    acc = Vector256.FusedMultiplyAdd(dv, dv, acc);
                }
                v = Vector256.Sum(acc);
                for (; i < cnt; i++)
                {
                    double dd = xb[i] - mean;
                    v += dd * dd;
                }
            }
            else
            {
                for (long i = 0; i < cnt; i++)
                {
                    double dd = xb[i] - mean;
                    v += dd * dd;
                }
            }

            float inv = 1f / MathF.Sqrt((float)(v / cnt) + _eps);
            for (long c = 0; c < cg; c++)
            {
                long ch = g * cg + c;
                float sc = (w == null ? 1f : w[ch]) * inv;
                float sh = (bi == null ? 0f : bi[ch]) - mean * sc;
                float* xr = xb + c * _s;
                float* yr = y + baseOff + c * _s;
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var sv = Vector256.Create(sc);
                    var hv = Vector256.Create(sh);
                    for (; i + 8 <= _s; i += 8)
                    {
                        Vector256.FusedMultiplyAdd(Vector256.Load(xr + i), sv, hv).Store(yr + i);
                    }
                }
                for (; i < _s; i++)
                {
                    yr[i] = xr[i] * sc + sh;
                }
            }
        }
    }
}
