using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>MatMulId: per (token, slot) matrix-vector product with the selected expert's [N, K] matrix.</summary>
internal sealed unsafe class MatMulIdOp : CpuOp
{
    const int RowsPerUnit = 8;
    readonly TRef _x, _w, _ids, _y;
    readonly WeightFormat _fmt;
    readonly long _n, _k, _rowBytes, _expertBytes;
    readonly int _experts, _slots;
    readonly bool _perSlot;
    readonly Action<int> _body;
    long _items, _units, _per, _groups;
    [ThreadStatic] static float[]? s_buf;
    public override string Name => $"MatMulId.{_fmt}";

    public MatMulIdOp(TRef x, TRef w, TRef ids, TRef y, WeightFormat fmt)
    {
        _x = x; _w = w; _ids = ids; _y = y; _fmt = fmt;
        _experts = (int)w.Dims[0]; _n = w.Dims[1]; _k = w.Dims[2];
        _slots = (int)ids.Dims[1];
        _perSlot = x.Rank == 3;
        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        _expertBytes = _rowBytes * _n;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long t = _ids.Dim(0, p);
        _items = t * _slots;
        if (_items == 0) return;
        _groups = (_n + RowsPerUnit - 1) / RowsPerUnit;
        _units = _items * _groups;
        int chunks = (int)Math.Min(_units, Par.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long u0 = c * _per, u1 = Math.Min(_units, u0 + _per);
        var buf = s_buf;
        if (buf is null || buf.Length < _k) s_buf = buf = new float[Math.Max(_k, 4096)];
        fixed (float* wbuf = buf)
        {
            for (long u = u0; u < u1; u++)
            {
                long item = u / _groups, g = u % _groups;
                int e = ((int*)_ids.P)[item];
                if ((uint)e >= (uint)_experts) throw new InvalidOperationException($"MatMulId: expert id {e} outside [0, {_experts}).");
                float* x = _x.F(_perSlot ? item : item / _slots);
                float* y = (float*)_y.P + item * _n;
                byte* wbase = _w.P + e * _expertBytes;
                long n0 = g * RowsPerUnit, n1 = Math.Min(_n, n0 + RowsPerUnit);
                for (long n = n0; n < n1; n++)
                {
                    byte* row = wbase + n * _rowBytes;
                    float* wr;
                    if (_fmt == WeightFormat.F32) wr = (float*)row;
                    else { Quant.Dequantize(_fmt, row, _k, wbuf); wr = wbuf; }
                    y[n] = Simd.Dot(x, wr, (int)_k);
                }
            }
        }
    }
}
