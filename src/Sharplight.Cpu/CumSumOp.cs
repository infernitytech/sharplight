using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Cumulative sum or product along one axis (optionally exclusive and/or reversed).</summary>
internal sealed unsafe class CumSumOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly bool _excl;
    private readonly bool _rev;
    private readonly bool _product;
    private readonly Action<int> _body;
    private long _outer;
    private long _a;
    private long _inner;
    private long _per;
    public override string Name => _product ? "CumProd" : "CumSum";
    public CumSumOp(TRef x, TRef y, int axis, bool exclusive, bool reverse, bool product = false)
    {
        _product = product;
        _x = x;
        _y = y;
        _axis = axis;
        _excl = exclusive;
        _rev = reverse;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _outer = OpHelpers.Prod(d, 0, _axis);
        _a = d[_axis];
        _inner = OpHelpers.Prod(d, _axis + 1, d.Length);
        if (_outer * _a * _inner == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_outer, _a * _inner);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_outer, s + _per);
        float* x = (float*)_x.P;
        float* y = (float*)_y.P;
        long n = _inner;
        for (long o = s; o < e; o++)
        {
            float* xb = x + o * _a * n;
            float* yb = y + o * _a * n;
            float* prev = null;
            for (long step = 0; step < _a; step++)
            {
                long a = _rev ? _a - 1 - step : step;
                float* xr = xb + a * n;
                float* yr = yb + a * n;
                if (prev == null)
                {
                    if (_excl)
                    {
                        new Span<float>(yr, (int)n).Fill(_product ? 1f : 0f);
                    }
                    else
                    {
                        Buffer.MemoryCopy(xr, yr, n * 4, n * 4);
                    }
                }
                else
                {
                    // inclusive: y[a] = y[prev] ∘ x[a];  exclusive: y[a] = y[prev] ∘ x[prev]
                    float* xs = _excl ? xb + (_rev ? a + 1 : a - 1) * n : xr;
                    if (_product)
                    {
                        for (long i = 0; i < n; i++)
                        {
                            yr[i] = prev[i] * xs[i];
                        }
                    }
                    else
                    {
                        for (long i = 0; i < n; i++)
                        {
                            yr[i] = prev[i] + xs[i];
                        }
                    }
                }
                prev = yr;
            }
        }
    }
}
