using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// MatMulId: items (token, slot) are grouped by expert; one unit computes four weight rows of one expert for all
/// of that expert's items, so every dequantized weight tile is reused across the rows (batched per expert).
/// </summary>
internal sealed unsafe class MatMulIdOp : CpuOp
{
    const int KB = 1024;
    readonly TRef _x, _w, _ids, _y;
    readonly WeightFormat _fmt;
    readonly long _n, _k, _rowBytes, _expertBytes;
    readonly int _experts, _slots;
    readonly bool _perSlot, _fused;
    readonly Action<int> _body;
    // Scratch sized for the largest item count: per-expert counts/offsets, sorted x and y row pointers, active experts.
    readonly int* _cnt, _start, _active, _cursor;
    readonly float** _xrows;
    readonly float** _yrows;
    long _units, _per, _groups;
    [ThreadStatic] static float[]? s_acc;
    public override string Name => $"MatMulId.{_fmt}";

    public MatMulIdOp(TRef x, TRef w, TRef ids, TRef y, WeightFormat fmt)
    {
        _x = x; _w = w; _ids = ids; _y = y; _fmt = fmt;
        _experts = (int)w.Dims[0]; _n = w.Dims[1]; _k = w.Dims[2];
        _slots = (int)ids.Dims[1];
        _perSlot = x.Rank == 3;
        if (_k % Quant.BlockSize(fmt) != 0) throw new ArgumentException($"K={_k} is not a multiple of the {fmt} block size.");
        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        _expertBytes = _rowBytes * _n;
        _fused = Quant.HasFusedKernel(fmt);
        long maxItems = Math.Max(1, ids.MaxElements);
        _cnt = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _start = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _active = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _cursor = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _xrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _yrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _body = Body;
    }

    public override void Dispose()
    {
        Mem.Free((byte*)_cnt); Mem.Free((byte*)_start); Mem.Free((byte*)_active); Mem.Free((byte*)_cursor); Mem.Free((byte*)_xrows); Mem.Free((byte*)_yrows);
    }

    public override void Run(RunParameters p)
    {
        long items = _ids.Dim(0, p) * _slots;
        if (items == 0) return;
        int* ids = (int*)_ids.P;
        new Span<int>(_cnt, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if ((uint)e >= (uint)_experts) throw new InvalidOperationException($"MatMulId: expert id {e} outside [0, {_experts}).");
            _cnt[e]++;
        }
        int off = 0, na = 0;
        for (int e = 0; e < _experts; e++)
        {
            _start[e] = off;
            if (_cnt[e] > 0) _active[na++] = e;
            off += _cnt[e];
        }
        new Span<int>(_cursor, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            long pos = _start[e] + _cursor[e]++;
            _xrows[pos] = _x.F(_perSlot ? i : i / _slots);
            _yrows[pos] = (float*)_y.P + i * _n;
        }
        _groups = (_n + 3) / 4;
        _units = na * _groups;
        int chunks = (int)Math.Min(_units, Par.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long u0 = c * _per, u1 = Math.Min(_units, u0 + _per);
        float* wbuf = stackalloc float[4 * KB];
        float* res = stackalloc float[8];
        float** wp = stackalloc float*[4];
        byte** rows = stackalloc byte*[4];
        int bs = Quant.BlockSize(_fmt), bb = Quant.BlockBytes(_fmt);
        for (long u = u0; u < u1; u++)
        {
            int e = _active[u / _groups];
            long g = u % _groups;
            long m = _cnt[e], m0 = _start[e];
            float** xr = _xrows + m0;
            float** yr = _yrows + m0;
            var accArr = s_acc;
            if (accArr is null || accArr.Length < m * 4) s_acc = accArr = new float[Math.Max(m * 4, 4096)];
            byte* wbase = _w.P + e * _expertBytes;
            long n0 = g * 4;
            int nr = (int)Math.Min(4, _n - n0);
            for (int r = 0; r < 4; r++) rows[r] = wbase + Math.Min(n0 + r, _n - 1) * _rowBytes;
            if (m == 1 && _fused)
            {
                Quant.DotRows4(_fmt, xr[0], rows, _k / bs, res);
                for (int r = 0; r < nr; r++) yr[0][n0 + r] = res[r];
                continue;
            }
            fixed (float* acc = accArr)
            {
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
                        Simd.Dot4x2(xr[t] + kb, xr[t + 1] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4, a1 = a0 + 4;
                        a0[0] += res[0]; a0[1] += res[1]; a0[2] += res[2]; a0[3] += res[3];
                        a1[0] += res[4]; a1[1] += res[5]; a1[2] += res[6]; a1[3] += res[7];
                    }
                    if (t < m)
                    {
                        Simd.Dot4(xr[t] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        a0[0] += res[0]; a0[1] += res[1]; a0[2] += res[2]; a0[3] += res[3];
                    }
                }
                for (long t = 0; t < m; t++)
                    for (int r = 0; r < nr; r++) yr[t][n0 + r] = acc[t * 4 + r];
            }
        }
    }
}
