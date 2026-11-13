using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>ONNX GatherElements: picks elements along one axis using an index tensor of the output shape.</summary>
internal sealed unsafe class GatherElementsOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _idx;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly Action<int> _body;
    private long[] _od = [];
    private long[] _xs = [];
    private long _total;
    private long _per;
    private long _axisN;
    public override string Name => "GatherElements";
    public GatherElementsOp(TRef x, TRef idx, TRef y, int axis)
    {
        _x = x;
        _idx = idx;
        _y = y;
        _axis = axis;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _od = OpHelpers.Dims(_idx, p);
        var xd = OpHelpers.Dims(_x, p);
        _xs = OpHelpers.Strides(xd);
        _axisN = xd[_axis];
        _total = OpHelpers.Prod(_od, 0, _od.Length);
        if (_total == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_total, 8);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        int r = _od.Length;
        long s = chunk * _per;
        long e = Math.Min(_total, s + _per);
        for (long l = s; l < e; l++)
        {
            long rem = l;
            long off = 0;
            for (int d = r - 1; d >= 0; d--)
            {
                long ix = rem % _od[d];
                rem /= _od[d];
                if (d == _axis)
                {
                    ix = OpHelpers.Wrap(OpHelpers.Idx(_idx, l), _axisN, "GatherElements");
                }

                off += ix * _xs[d];
            }
            OpHelpers.Cp(_y.P + l * es, _x.P + off * es, es);
        }
    }
}
