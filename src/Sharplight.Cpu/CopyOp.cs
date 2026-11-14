using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Materializes (possibly strided) data into a dense destination; also implements Contiguous. Large copies run in
/// parallel, split along the outer axis or, when it is short (batch 1), along the largest other axis.
/// </summary>
internal sealed unsafe class CopyOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly Action<int> _body;
    private long _chunkRows;
    private long _d0;
    private int _split;
    private long _splitDim;
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
            int want = CpuParallel.Chunks(total, 32768);
            _split = 0;
            _splitDim = d0;
            for (int k = 1; k < _x.Rank && _splitDim < want; k++)
            {
                if (_x.Dims[k] > _splitDim)
                {
                    _split = k;
                    _splitDim = _x.Dims[k];
                }
            }

            int chunks = (int)Math.Min(_splitDim, want);
            _chunkRows = (_splitDim + chunks - 1) / chunks;
            CpuParallel.For(chunks, _body);
        }
        else
        {
            _split = 0;
            _splitDim = d0;
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
        long e = Math.Min(_splitDim, s + _chunkRows);
        if (s >= e)
        {
            return;
        }

        int es = _x.ElemSize;
        byte* src = _x.P + s * _x.Strides[_split] * es;
        byte* dst = _y.P + s * _y.Strides[_split] * es;
        if (_split == 0)
        {
            Mem.CopyStrided(src, _x.Strides, dst, _y.Strides, _x.Dims, e - s, es);
            return;
        }

        var dims = (long[])_x.Dims.Clone();
        dims[_split] = e - s;
        Mem.CopyStrided(src, _x.Strides, dst, _y.Strides, dims, _d0, es);
    }
}
