using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Materializes (possibly strided) data into a dense destination; also implements Contiguous.</summary>
internal sealed unsafe class CopyOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly Action<int> _body;
    private long _chunkRows;
    private long _d0;
    public override string Name => "Copy";
    public CopyOp(TRef x, TRef y)
    {
        _x = x;
        _y = y;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long d0 = _x.Rank == 0 ? 1 : _x.Dim(0, p);
        _d0 = d0;
        long total = _x.Elements(p);
        if (total == 0)
        {
            return;
        }

        if (_x.Rank >= 2 && total > 65536)
        {
            int chunks = (int)Math.Min(d0, CpuParallel.Chunks(total, 32768));
            _chunkRows = (d0 + chunks - 1) / chunks;
            CpuParallel.For(chunks, _body);
        }
        else
        {
            _chunkRows = d0;
            Body(0);
        }
    }

    private void Body(int c)
    {
        if (_x.Rank == 0)
        {
            Mem.CopyStrided(_x.P, _x.Strides, _y.P, _y.Strides, _x.Dims, 1, _x.ElemSize);
            return;
        }
        long s = c * _chunkRows;
        long e = Math.Min(_d0, s + _chunkRows);
        if (s >= e)
        {
            return;
        }

        int es = _x.ElemSize;
        Mem.CopyStrided(_x.P + s * _x.Strides[0] * es, _x.Strides, _y.P + s * _y.Strides[0] * es, _y.Strides, _x.Dims, e - s, es);
    }
}
