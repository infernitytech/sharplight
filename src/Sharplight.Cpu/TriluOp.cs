using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Triangular masking of the last two axes (any plain element type).</summary>
internal sealed unsafe class TriluOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _upper;
    private readonly long _k;
    private readonly Action<int> _body;
    private long _batch;
    private long _m;
    private long _n;
    private long _per;
    public override string Name => "Trilu";
    public TriluOp(TRef x, TRef y, bool upper, long k)
    {
        _x = x;
        _y = y;
        _upper = upper;
        _k = k;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _m = d[^2];
        _n = d[^1];
        _batch = OpHelpers.Prod(d, 0, d.Length - 2);
        if (_batch * _m * _n == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_batch * _m, _n);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per;
        long e = Math.Min(_batch * _m, s + _per);
        for (long r = s; r < e; r++)
        {
            long i = r % _m;
            byte* src = _x.P + r * _n * es;
            byte* dst = _y.P + r * _n * es;
            for (long j = 0; j < _n; j++)
            {
                bool keep = _upper ? j - i >= _k : j - i <= _k;
                if (keep)
                {
                    OpHelpers.Cp(dst + j * es, src + j * es, es);
                }
                else
                {
                    new Span<byte>(dst + j * es, es).Clear();
                }
            }
        }
    }
}
