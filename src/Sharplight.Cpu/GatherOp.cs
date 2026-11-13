using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Row gather from a (possibly quantized) table; indices are i32.</summary>
internal sealed unsafe class GatherOp : CpuOp
{
    private readonly TRef _table;
    private readonly TRef _idx;
    private readonly TRef _y;
    private readonly WeightFormat _fmt;
    private readonly long _v;
    private readonly long _d;
    private readonly long _rowBytes;
    private readonly Action<int> _body;
    private long _n;
    private long _per;
    public override string Name => "Gather";

    public GatherOp(TRef table, TRef idx, TRef y, WeightFormat fmt)
    {
        _table = table;
        _idx = idx;
        _y = y;
        _fmt = fmt;
        _v = table.Dims[0];
        _d = 1;
        for (int i = 1; i < table.Rank; i++)
        {
            _d *= table.Dims[i];
        }

        _rowBytes = _d / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _n = _idx.Dim(0, p);
        if (_n == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_n, CpuParallel.Chunks(_n * _d, 8192));
        _per = (_n + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_n, s + _per);
        for (long i = s; i < e; i++)
        {
            long id = ((int*)_idx.P)[i * _idx.Strides[0]];
            if ((ulong)id >= (ulong)_v)
            {
                throw new ArgumentOutOfRangeException("indices", $"Gather index {id} out of range [0,{_v}).");
            }

            byte* src = _table.P + id * _rowBytes;
            if (_y.Type == DataType.F32)
            {
                Quant.Dequantize(_fmt, src, _d, (float*)_y.P + i * _d);
            }
            else
            {
                Buffer.MemoryCopy(src, _y.P + i * _d * _y.ElemSize, _d * _y.ElemSize, _d * _y.ElemSize);
            }
        }
    }
}
