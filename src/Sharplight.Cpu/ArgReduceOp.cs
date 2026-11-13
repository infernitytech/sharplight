using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Index of the maximum or minimum along one axis (ONNX ArgMax/ArgMin semantics, optionally selecting the last tie).</summary>
internal sealed unsafe class ArgReduceOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly bool _max;
    private readonly bool _last;
    private readonly Action<int> _body;
    private long _outer;
    private long _a;
    private long _inner;
    private long _per;
    public override string Name => "ArgReduce." + (_max ? "max" : "min");
    public ArgReduceOp(TRef x, TRef y, int axis, bool max, bool selectLast)
    {
        _x = x;
        _y = y;
        _axis = axis;
        _max = max;
        _last = selectLast;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _outer = OpHelpers.Prod(d, 0, _axis);
        _a = d[_axis];
        _inner = OpHelpers.Prod(d, _axis + 1, d.Length);
        long items = _outer * _inner;
        if (items == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(items, _a);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_outer * _inner, s + _per);
        float* x = (float*)_x.P;
        int* y = (int*)_y.P;
        for (long it = s; it < e; it++)
        {
            long o = it / _inner;
            long i = it % _inner;
            float* b = x + o * _a * _inner + i;
            int best = 0;
            float bv = b[0];
            for (int a = 1; a < _a; a++)
            {
                float v = b[a * _inner];
                bool better = _max ? (_last ? v >= bv : v > bv) : (_last ? v <= bv : v < bv);
                if (better)
                {
                    bv = v;
                    best = a;
                }
            }
            y[it] = best;
        }
    }
}
