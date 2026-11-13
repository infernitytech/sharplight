using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>y = x · Wᵀ with on-the-fly dequantization of W. x rows are f32; W is [N, K].</summary>
internal sealed unsafe class MatMulOp : CpuOp
{
    const int KB = 1024;
    readonly TRef _x, _w, _y;
    readonly WeightFormat _fmt;
    readonly long _n, _k, _rowBytes;
    readonly bool _fused;
    readonly float** _xrows;
    readonly Action<int> _body;
    [ThreadStatic] static float[]? s_acc;
    long _m, _groups, _perChunk;
    public override string Name => $"MatMul.{_fmt}";

    public MatMulOp(TRef x, TRef w, TRef y, WeightFormat fmt)
    {
        _x = x; _w = w; _y = y; _fmt = fmt;
        _n = w.Dims[0]; _k = w.Dims[1];
        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        if (_k % Quant.BlockSize(fmt) != 0) throw new ArgumentException($"K={_k} is not a multiple of the {fmt} block size.");
        long maxRows = Math.Max(1, x.MaxElements / Math.Max(1, _k));
        _fused = Quant.HasFusedKernel(fmt);
        _xrows = (float**)Mem.Alloc(maxRows * sizeof(float*), false);
        _body = Body;
    }

    public override void Dispose() => Mem.Free((byte*)_xrows);

    public override void Run(RunParameters p)
    {
        _m = _x.Rows(p);
        if (_m == 0) return;
        for (long r = 0; r < _m; r++) _xrows[r] = _x.F(r);
        _groups = (_n + 3) / 4;
        int chunks = (int)Math.Min(_groups, Par.Threads * 8L);
        _perChunk = (_groups + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long g0 = c * _perChunk, g1 = Math.Min(_groups, g0 + _perChunk);
        long m = _m;
        var accArr = s_acc;
        if (accArr is null || accArr.Length < m * 4) s_acc = accArr = new float[Math.Max(m * 4, 4096)];
        float* wbuf = stackalloc float[4 * KB];
        float* res = stackalloc float[8];
        float** wp = stackalloc float*[4];
        byte** rows = stackalloc byte*[4];
        int bs = Quant.BlockSize(_fmt), bb = Quant.BlockBytes(_fmt);
        fixed (float* acc = accArr)
        {
            for (long g = g0; g < g1; g++)
            {
                long n0 = g * 4;
                int nr = (int)Math.Min(4, _n - n0);
                for (int r = 0; r < 4; r++) rows[r] = _w.P + Math.Min(n0 + r, _n - 1) * _rowBytes;
                if (m == 1 && _fused)
                {
                    Quant.DotRows4(_fmt, _xrows[0], rows, _k / bs, res);
                    float* y1 = (float*)_y.P + n0;
                    for (int r = 0; r < nr; r++) y1[r] = res[r];
                    continue;
                }
                new Span<float>(acc, (int)(m * 4)).Clear();
                for (long kb = 0; kb < _k; kb += KB)
                {
                    long len = Math.Min(KB, _k - kb);
                    for (int r = 0; r < 4; r++)
                    {
                        if (_fmt == WeightFormat.F32) wp[r] = (float*)rows[r] + kb;
                        else
                        {
                            Quant.Dequantize(_fmt, rows[r] + kb / bs * bb, len, wbuf + r * KB);
                            wp[r] = wbuf + r * KB;
                        }
                    }
                    long t = 0;
                    for (; t + 2 <= m; t += 2)
                    {
                        Simd.Dot4x2(_xrows[t] + kb, _xrows[t + 1] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4, a1 = a0 + 4;
                        a0[0] += res[0]; a0[1] += res[1]; a0[2] += res[2]; a0[3] += res[3];
                        a1[0] += res[4]; a1[1] += res[5]; a1[2] += res[6]; a1[3] += res[7];
                    }
                    if (t < m)
                    {
                        Simd.Dot4(_xrows[t] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        a0[0] += res[0]; a0[1] += res[1]; a0[2] += res[2]; a0[3] += res[3];
                    }
                }
                for (long t = 0; t < m; t++)
                {
                    float* y = (float*)_y.P + t * _n + n0;
                    for (int r = 0; r < nr; r++) y[r] = acc[t * 4 + r];
                }
            }
        }
    }
}

/// <summary>Row gather from a (possibly quantized) table; indices are i32.</summary>
internal sealed unsafe class GatherOp : CpuOp
{
    readonly TRef _table, _idx, _y; readonly WeightFormat _fmt;
    readonly long _v, _d, _rowBytes;
    readonly Action<int> _body;
    long _n, _per;
    public override string Name => "Gather";

    public GatherOp(TRef table, TRef idx, TRef y, WeightFormat fmt)
    {
        _table = table; _idx = idx; _y = y; _fmt = fmt;
        _v = table.Dims[0];
        _d = 1; for (int i = 1; i < table.Rank; i++) _d *= table.Dims[i];
        _rowBytes = _d / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _n = _idx.Dim(0, p);
        if (_n == 0) return;
        int chunks = (int)Math.Min(_n, Par.Chunks(_n * _d, 8192));
        _per = (_n + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long s = c * _per, e = Math.Min(_n, s + _per);
        for (long i = s; i < e; i++)
        {
            long id = ((int*)_idx.P)[i * _idx.Strides[0]];
            if ((ulong)id >= (ulong)_v) throw new ArgumentOutOfRangeException("indices", $"Gather index {id} out of range [0,{_v}).");
            byte* src = _table.P + id * _rowBytes;
            if (_y.Type == DataType.F32) Quant.Dequantize(_fmt, src, _d, (float*)_y.P + i * _d);
            else Buffer.MemoryCopy(src, _y.P + i * _d * _y.ElemSize, _d * _y.ElemSize, _d * _y.ElemSize);
        }
    }
}

internal sealed unsafe class ConcatOp : CpuOp
{
    readonly TRef[] _xs; readonly TRef _y; readonly int _axis;
    readonly long[] _dstStrides; readonly long[] _offsets;
    public override string Name => "Concat";

    public ConcatOp(TRef[] xs, TRef y, int axis)
    {
        _xs = xs; _y = y; _axis = axis;
        _dstStrides = TRef.RowMajor(y.Dims);
        _offsets = new long[xs.Length];
        long o = 0;
        for (int i = 0; i < xs.Length; i++) { _offsets[i] = o; o += xs[i].Dims[axis]; }
    }

    public override void Run(RunParameters p)
    {
        int es = _y.ElemSize;
        for (int i = 0; i < _xs.Length; i++)
        {
            var x = _xs[i];
            Mem.CopyStrided(x.P, x.Strides, _y.P + _offsets[i] * _dstStrides[_axis] * es, _dstStrides, x.Dims, x.Dim(0, p), es);
        }
    }
}
