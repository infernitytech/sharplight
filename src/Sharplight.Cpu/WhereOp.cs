using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>y = cond ? a : b with broadcasting; any plain element type for a/b (copied by element size).</summary>
internal sealed unsafe class WhereOp : CpuOp
{
    private readonly TRef _c;
    private readonly TRef _a;
    private readonly TRef _b;
    private readonly TRef _y;
    private readonly long[] _sc;
    private readonly long[] _sa;
    private readonly long[] _sb;
    private readonly Action<int> _body;
    private long _outer;
    private long _per;
    private long _l;
    public override string Name => "Where";

    public WhereOp(TRef c, TRef a, TRef b, TRef y)
    {
        _c = c;
        _a = a;
        _b = b;
        _y = y;
        _body = Body;
        _sc = OpHelpers.Align(c, y.Rank);
        _sa = OpHelpers.Align(a, y.Rank);
        _sb = OpHelpers.Align(b, y.Rank);
    }

    private bool Cond(long off)
    {
        var t = _c.Type;
        if (t == DataType.F32)
        {
            return ((float*)_c.P)[off] != 0f;
        }

        if (t == DataType.I32)
        {
            return ((int*)_c.P)[off] != 0;
        }

        if (t == DataType.I64)
        {
            return ((long*)_c.P)[off] != 0;
        }

        return _c.P[off] != 0;
    }

    public override void Run(RunParameters p)
    {
        long total = _y.Elements(p);
        if (total == 0)
        {
            return;
        }

        _l = _y.Cols(p);
        _outer = total / _l;
        var (ch, per) = OpHelpers.Split(_outer, _l);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int r = _y.Rank;
        int es = _y.ElemSize;
        long s = chunk * _per;
        long e = Math.Min(_outer, s + _per);
        for (long row = s; row < e; row++)
        {
            long oc = 0;
            long oa = 0;
            long ob = 0;
            long rr = row;
            for (int k = r - 2; k >= 0; k--)
            {
                long d = k == 0 && _y.Sym is not null ? _y.Dims[0] : _y.Dims[k];
                long ix = rr % d;
                rr /= d;
                oc += ix * _sc[k];
                oa += ix * _sa[k];
                ob += ix * _sb[k];
            }
            long sc = _sc[r - 1];
            long sa = _sa[r - 1];
            long sb = _sb[r - 1];
            byte* y = _y.P + row * _l * es;
            for (long i = 0; i < _l; i++)
            {
                OpHelpers.Cp(y + i * es, (Cond(oc + i * sc) ? _a.P + (oa + i * sa) * es : _b.P + (ob + i * sb) * es), es);
            }
        }
    }
}
