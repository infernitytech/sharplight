using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>One-hot encoding of an index tensor along a new axis with configurable on/off values.</summary>
internal sealed unsafe class OneHotOp : CpuOp
{
    private readonly TRef _idx;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly long _depth;
    private readonly float _on;
    private readonly float _off;
    private readonly Action<int> _body;
    private long _pre;
    private long _post;
    private long _per;
    public override string Name => "OneHot";
    public OneHotOp(TRef idx, TRef y, int axis, long depth, float on, float off)
    {
        _idx = idx;
        _y = y;
        _axis = axis;
        _depth = depth;
        _on = on;
        _off = off;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_idx, p);
        _pre = OpHelpers.Prod(d, 0, _axis);
        _post = OpHelpers.Prod(d, _axis, d.Length);
        if (_pre * _post * _depth == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_pre, _post * _depth);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_pre, s + _per);
        float* y = (float*)_y.P;
        for (long pr = s; pr < e; pr++)
        {
            float* yb = y + pr * _depth * _post;
            new Span<float>(yb, (int)(_depth * _post)).Fill(_off);
            for (long q = 0; q < _post; q++)
            {
                long id = OpHelpers.Idx(_idx, pr * _post + q);
                if (id < 0)
                {
                    id += _depth;
                }

                if ((ulong)id < (ulong)_depth)
                {
                    yb[id * _post + q] = _on;
                }
            }
        }
    }
}
