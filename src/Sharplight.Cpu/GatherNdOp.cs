using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>ONNX GatherND: gathers slices addressed by index tuples, with optional leading batch dimensions.</summary>
internal sealed unsafe class GatherNdOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _idx;
    private readonly TRef _y;
    private readonly int _b;
    private readonly Action<int> _body;
    private long[] _xd = [];
    private long[] _xs = [];
    private long _tuples;
    private long _q;
    private long _slice;
    private long _perBatch;
    private long _tuplesPerBatch;
    private long _per;
    public override string Name => "GatherND";
    public GatherNdOp(TRef x, TRef idx, TRef y, int batchDims)
    {
        _x = x;
        _idx = idx;
        _y = y;
        _b = batchDims;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var id = OpHelpers.Dims(_idx, p);
        _xd = OpHelpers.Dims(_x, p);
        _xs = OpHelpers.Strides(_xd);
        _q = id[^1];
        _tuples = OpHelpers.Prod(id, 0, id.Length - 1);
        _tuplesPerBatch = OpHelpers.Prod(id, _b, id.Length - 1);
        _perBatch = OpHelpers.Prod(_xd, _b, _xd.Length);
        _slice = OpHelpers.Prod(_xd, _b + (int)_q, _xd.Length);
        if (_tuples == 0 || _slice == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(_tuples, _slice);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per;
        long e = Math.Min(_tuples, s + _per);
        for (long t = s; t < e; t++)
        {
            long batch = t / _tuplesPerBatch;
            long off = batch * _perBatch;
            for (int j = 0; j < _q; j++)
            {
                off += OpHelpers.Wrap(OpHelpers.Idx(_idx, t * _q + j), _xd[_b + j], "GatherND") * _xs[_b + j];
            }

            Buffer.MemoryCopy(_x.P + off * es, _y.P + t * _slice * es, _slice * es, _slice * es);
        }
    }
}
