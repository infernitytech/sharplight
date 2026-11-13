using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>numpy take along an axis: out[o, q, i] = data[o, idx[q], i].</summary>
internal sealed unsafe class GatherAxisOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _idx;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly Action<int> _body;
    private long _outer;
    private long _a;
    private long _inner;
    private long _q;
    private long _per;
    public override string Name => "GatherAxis";
    public GatherAxisOp(TRef x, TRef idx, TRef y, int axis)
    {
        _x = x;
        _idx = idx;
        _y = y;
        _axis = axis;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _outer = OpHelpers.Prod(d, 0, _axis);
        _a = d[_axis];
        _inner = OpHelpers.Prod(d, _axis + 1, d.Length);
        _q = _idx.Elements(p);
        long items = _outer * _q;
        if (items == 0 || _inner == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(items, _inner);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per;
        long e = Math.Min(_outer * _q, s + _per);
        for (long it = s; it < e; it++)
        {
            long o = it / _q;
            long q = it % _q;
            long id = OpHelpers.Wrap(OpHelpers.Idx(_idx, q), _a, "GatherAxis");
            Buffer.MemoryCopy(_x.P + ((o * _a + id) * _inner) * es, _y.P + it * _inner * es, _inner * es, _inner * es);
        }
    }
}
