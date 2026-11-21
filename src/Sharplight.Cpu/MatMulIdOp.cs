using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// MatMulId: items (token, slot) are grouped by expert; one unit computes four weight rows of one expert for all
/// of that expert's items, so every dequantized weight tile is reused across the rows (batched per expert). Two products over the same
/// rows and ids (an MLP's gate and up projections) can run as one op: one quantization, one sort, one parallel pass.
/// </summary>
/// <remarks>
/// With integer dots (<see cref="Runtime.DeviceFeatures.IntDot"/> enabled and the format decodable, <see cref="IntDot"/>) the activation
/// rows are quantized to 8 bits once and a unit decodes a run of weight rows of one expert, each once, for all of that expert's items;
/// every (row, item) product is computed alone, so the result does not depend on how many items an expert has.
/// An expert id of -1 selects no expert: the item's output rows are zero (the slot runs on another device).
/// </remarks>
internal sealed unsafe class MatMulIdOp : CpuOp
{
    private const int KB = 1024;
    /// <summary>The expert id of an item that selects no expert (its output rows are zero; see <c>CompileOptions.SparseExpertSlots</c>).</summary>
    private const int NoExpert = -1;

    /// <summary>
    /// Up to this many rows every row is computed alone, as a single row is (the fused dequantize-dot kernel once per row, or the
    /// one-row dot instead of the two-row one), so a speculative verification step (2-4 rows) gives the logits of plain one-row decoding
    /// bit for bit.
    /// </summary>
    private const int FusedRowsMax = 4;

    /// <summary>Weight rows a unit of the integer-dot path decodes at most, and fewer when the units would not keep every thread busy.</summary>
    private const int IntDotRowsPerUnit = 32;

    /// <summary>Weight rows prefetched ahead of the one being decoded.</summary>
    private const int PrefetchRows = 2;
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _ids;
    private readonly TRef _y;
    /// <summary>Optional second product over the same rows and ids (an MLP's gate and up projections): weight and output, or null.</summary>
    private readonly TRef? _w2;
    private readonly TRef? _y2;
    private readonly int _products;
    private readonly WeightFormat _fmt;
    private readonly long _n;
    private readonly long _k;
    private readonly long _rowBytes;
    private readonly long _expertBytes;
    private readonly int _experts;
    private readonly int _slots;
    private readonly bool _perSlot;
    private readonly bool _fused;
    private readonly bool _intDot;
    private readonly bool _smallCodes;
    private readonly bool _fusedDot;
    private readonly Action<int> _body;
    private readonly Action<int> _intDotBody;
    private readonly Action<int> _quantizeBody;
    // Scratch sized for the largest item count: per-expert counts/offsets, sorted x and y row pointers, active experts.
    private readonly int* _cnt;
    private readonly int* _start;
    private readonly int* _active;
    private readonly int* _cursor;
    private readonly float** _xrows;
    private readonly float** _yrows;
    private readonly float** _yrows2;
    // Integer-dot path: the quantized activation row of every sorted item, the quantized rows and the rows per unit.
    private readonly byte** _qrows;
    private byte* _activations;
    private long _activationRows;
    private long _rowsPerUnit;
    private long _units;
    /// <summary>Units per product (the units of the second product follow those of the first).</summary>
    private long _unitsPerProduct;
    private long _per;
    private long _groups;
    [ThreadStatic] private static float[]? s_acc;
    public override string Name => _products == 2 ? $"MatMulId.{_fmt} x2" : $"MatMulId.{_fmt}";

    /// <param name="x">Rows [T, K] or [T, S, K].</param>
    /// <param name="w">Expert weights [E, N, K].</param>
    /// <param name="ids">Expert per (row, slot) [T, S].</param>
    /// <param name="y">Output [T, S, N].</param>
    /// <param name="fmt">Weight format.</param>
    /// <param name="intDot">Integer dots with 8-bit activations where the format has them.</param>
    /// <param name="w2">A second weight of the same shape and format over the same rows and ids (computed in the same pass), or null.</param>
    /// <param name="y2">The second product's output.</param>
    public MatMulIdOp(TRef x, TRef w, TRef ids, TRef y, WeightFormat fmt, bool intDot = false, TRef? w2 = null, TRef? y2 = null)
    {
        _x = x;
        _w = w;
        _ids = ids;
        _y = y;
        _w2 = w2;
        _y2 = y2;
        _products = w2 is null ? 1 : 2;
        _fmt = fmt;
        _experts = (int)w.Dims[0];
        _n = w.Dims[1];
        _k = w.Dims[2];
        _slots = (int)ids.Dims[1];
        _perSlot = x.Rank == 3;
        if (_k % Quant.BlockSize(fmt) != 0)
        {
            throw new ArgumentException($"K={_k} is not a multiple of the {fmt} block size.");
        }

        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        _expertBytes = _rowBytes * _n;
        _fused = Quant.HasFusedKernel(fmt);
        _intDot = intDot && IntDot.Supports(fmt) && _k % IntDot.BlockSize == 0 && _k <= int.MaxValue;
        _smallCodes = IntDot.HasSmallCodes(fmt);
        _fusedDot = _intDot && IntDot.HasFusedDot(fmt) && _k % Quant.BlockSize(fmt) == 0;
        long maxItems = Math.Max(1, ids.MaxElements);
        _cnt = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _start = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _active = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _cursor = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _xrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _yrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _yrows2 = w2 is null ? null : (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _qrows = (byte**)Mem.Alloc(maxItems * sizeof(byte*), false);
        _body = Body;
        _intDotBody = IntDotBody;
        _quantizeBody = QuantizeBody;
    }

    public override void Dispose()
    {
        Mem.Free((byte*)_cnt);
        Mem.Free((byte*)_start);
        Mem.Free((byte*)_active);
        Mem.Free((byte*)_cursor);
        Mem.Free((byte*)_xrows);
        Mem.Free((byte*)_yrows);
        if (_yrows2 != null)
        {
            Mem.Free((byte*)_yrows2);
        }
        Mem.Free((byte*)_qrows);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        long items = _ids.Dim(0, p) * _slots;
        if (items == 0)
        {
            return;
        }

        int* ids = (int*)_ids.P;
        new Span<int>(_cnt, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e == NoExpert)
            {
                // The item selects no expert (its slot runs elsewhere): its products are zero.
                new Span<float>((float*)_y.P + i * _n, (int)_n).Clear();
                if (_y2 is { } y2)
                {
                    new Span<float>((float*)y2.P + i * _n, (int)_n).Clear();
                }

                continue;
            }

            if ((uint)e >= (uint)_experts)
            {
                throw new InvalidOperationException($"MatMulId: expert id {e} outside [0, {_experts}).");
            }

            _cnt[e]++;
        }
        int off = 0;
        int na = 0;
        for (int e = 0; e < _experts; e++)
        {
            _start[e] = off;
            if (_cnt[e] > 0)
            {
                _active[na++] = e;
            }

            off += _cnt[e];
        }

        if (na == 0)
        {
            return;
        }
        new Span<int>(_cursor, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e == NoExpert)
            {
                continue;
            }

            long pos = _start[e] + _cursor[e]++;
            _xrows[pos] = _x.F(_perSlot ? i : i / _slots);
            _yrows[pos] = (float*)_y.P + i * _n;
            if (_yrows2 != null)
            {
                _yrows2[pos] = (float*)_y2!.P + i * _n;
            }
        }

        if (_intDot)
        {
            // Fused dots recompute the grid lookups for every item: experts with many items (prefill) dequantize their tiles once on the
            // float path instead, after the experts with few items ran on the integer dots.
            int many = _fusedDot ? PartitionByItems(na) : 0;
            if (many < na)
            {
                RunIntDot(ids, items, na - many);
            }

            if (many == 0)
            {
                return;
            }

            // The float path's experts follow the integer-dot ones in _active.
            new Span<int>(_active + (na - many), many).CopyTo(new Span<int>(_active, many));
            na = many;
        }

        _groups = (_n + 3) / 4;
        _unitsPerProduct = na * _groups;
        _units = _unitsPerProduct * _products;
        int chunks = (int)Math.Min(_units, CpuParallel.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    /// <summary>
    /// Orders <see cref="_active"/> so that the experts with at most <see cref="FusedRowsMax"/> items come first (in expert order, which
    /// keeps every row's result the same whatever the other experts' counts); returns the number of experts with more.
    /// </summary>
    private int PartitionByItems(int activeExperts)
    {
        int few = 0;
        int many = 0;
        int* manyExperts = _cursor;   // free scratch until RunIntDot fills its cursors
        for (int i = 0; i < activeExperts; i++)
        {
            int e = _active[i];
            if (_cnt[e] <= FusedRowsMax)
            {
                _active[few++] = e;
            }
            else
            {
                manyExperts[many++] = e;
            }
        }

        new Span<int>(manyExperts, many).CopyTo(new Span<int>(_active + few, many));
        return many;
    }

    /// <summary>Quantizes every distinct activation row once, points each sorted item at its row and runs the units.</summary>
    private void RunIntDot(int* ids, long items, int activeExperts)
    {
        long rowBytes = IntDot.RowBytes(_k);
        _activationRows = _perSlot ? items : items / _slots;
        _activations = ThreadScratch.Get(ThreadScratch.Caller, _activationRows * rowBytes);
        if (_activationRows > 1)
        {
            CpuParallel.For((int)Math.Min(_activationRows, CpuParallel.Threads * 2L), _quantizeBody);
        }
        else
        {
            IntDot.Quantize(_x.F(0), _k, IntDot.RowAt(_activations, _k));
        }

        new Span<int>(_cursor, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e == NoExpert)
            {
                continue;
            }

            long pos = _start[e] + _cursor[e]++;
            _qrows[pos] = _activations + (_perSlot ? i : i / _slots) * rowBytes;
        }

        // Enough units for every thread, each a run of rows of one expert.
        long target = CpuParallel.Threads * 4L;
        long perExpert = Math.Max(1, (target + activeExperts * _products - 1) / (activeExperts * _products));
        _rowsPerUnit = Math.Clamp((_n + perExpert - 1) / perExpert, 1, IntDotRowsPerUnit);
        _groups = (_n + _rowsPerUnit - 1) / _rowsPerUnit;
        _unitsPerProduct = activeExperts * _groups;
        _units = _unitsPerProduct * _products;
        int chunks = (int)Math.Min(_units, CpuParallel.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        CpuParallel.For(chunks, _intDotBody);
    }

    private void QuantizeBody(int c)
    {
        int chunks = (int)Math.Min(_activationRows, CpuParallel.Threads * 2L);
        long per = (_activationRows + chunks - 1) / chunks;
        long rowBytes = IntDot.RowBytes(_k);
        for (long r = c * per; r < Math.Min(_activationRows, (c + 1) * per); r++)
        {
            IntDot.Quantize(_x.F(r), _k, IntDot.RowAt(_activations + r * rowBytes, _k));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void IntDotBody(int c)
    {
        long u0 = c * _per;
        long u1 = Math.Min(_units, u0 + _per);
        int k = (int)_k;
        var level = CpuIsa.Level;
        byte* u = ThreadScratch.Get(ThreadScratch.Worker, IntDot.DecodedBytes(_k));
        float* a = (float*)(u + ((_k + 63) & ~63L));
        float* b = a + ((_k / IntDot.GroupSize + 15) & ~15L);
        for (long unit = u0; unit < u1; unit++)
        {
            bool second = unit >= _unitsPerProduct;
            long local = second ? unit - _unitsPerProduct : unit;
            int e = _active[local / _groups];
            long n0 = local % _groups * _rowsPerUnit;
            long n1 = Math.Min(_n, n0 + _rowsPerUnit);
            long m = _cnt[e];
            byte** xr = _qrows + _start[e];
            float** yr = (second ? _yrows2 : _yrows) + _start[e];
            byte* wbase = (second ? _w2!.P : _w.P) + e * _expertBytes;
            for (long n = n0; n < n1; n++)
            {
                if (n + PrefetchRows < n1)
                {
                    Simd.Prefetch(wbase + (n + PrefetchRows) * _rowBytes, _rowBytes);
                }

                byte* row = wbase + n * _rowBytes;
                if (_fusedDot)
                {
                    for (long t = 0; t < m; t++)
                    {
                        yr[t][n] = IntDot.FusedDot(_fmt, row, IntDot.RowAt(xr[t], _k), k);
                    }

                    continue;
                }

                IntDot.Decode(_fmt, row, _k, u, a, b);
                for (long t = 0; t < m; t++)
                {
                    yr[t][n] = IntDot.Dot(level, u, a, b, _smallCodes, IntDot.RowAt(xr[t], _k), k);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int c)
    {
        long u0 = c * _per;
        long u1 = Math.Min(_units, u0 + _per);
        float* wbuf = stackalloc float[4 * KB];
        float* res = stackalloc float[8];
        float** wp = stackalloc float*[4];
        byte** rows = stackalloc byte*[4];
        int bs = Quant.BlockSize(_fmt);
        int bb = Quant.BlockBytes(_fmt);
        for (long u = u0; u < u1; u++)
        {
            bool second = u >= _unitsPerProduct;
            long local = second ? u - _unitsPerProduct : u;
            int e = _active[local / _groups];
            long g = local % _groups;
            long m = _cnt[e];
            long m0 = _start[e];
            float** xr = _xrows + m0;
            float** yr = (second ? _yrows2 : _yrows) + m0;
            var accArr = s_acc;
            if (accArr is null || accArr.Length < m * 4)
            {
                s_acc = accArr = new float[Math.Max(m * 4, 4096)];
            }

            byte* wbase = (second ? _w2!.P : _w.P) + e * _expertBytes;
            long n0 = g * 4;
            int nr = (int)Math.Min(4, _n - n0);
            for (int r = 0; r < 4; r++)
            {
                rows[r] = wbase + Math.Min(n0 + r, _n - 1) * _rowBytes;
            }

            if (m <= FusedRowsMax && _fused)
            {
                for (long t = 0; t < m; t++)
                {
                    Quant.DotRows4(_fmt, xr[t], rows, _k / bs, res);
                    for (int r = 0; r < nr; r++)
                    {
                        yr[t][n0 + r] = res[r];
                    }
                }

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
                        if (_fmt == WeightFormat.F32)
                        {
                            wp[r] = (float*)rows[r] + kb;
                        }
                        else
                        {
                            Quant.Dequantize(_fmt, rows[r] + kb / bs * bb, len, wbuf + r * KB);
                            wp[r] = wbuf + r * KB;
                        }
                    }
                    long t = 0;
                    for (; m > FusedRowsMax && t + 2 <= m; t += 2)
                    {
                        Simd.Dot4x2(xr[t] + kb, xr[t + 1] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        float* a1 = a0 + 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                        a1[0] += res[4];
                        a1[1] += res[5];
                        a1[2] += res[6];
                        a1[3] += res[7];
                    }
                    for (; t < m; t++)
                    {
                        Simd.Dot4(xr[t] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                    }
                }
                for (long t = 0; t < m; t++)
                {
                    for (int r = 0; r < nr; r++)
                    {
                        yr[t][n0 + r] = acc[t * 4 + r];
                    }
                }
            }
        }
    }
}
