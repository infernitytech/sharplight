using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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
/// A gate / up pair can also write the gated activation <c>SiLU(gate) * up</c> of an MLP (the expert layer's next two nodes): the integer-dot
/// units compute it for their rows right after the products, so a decode step runs one parallel pass instead of three.
/// A single product can also write the weighted sum over the slots of an MoE layer's output, <c>sum_s w[t, s] * y[t, s, :]</c> (the
/// layer's next three nodes: the Mul by the slot weights, the Permute and the ReduceSum), added in slot order like the separate ops: up to
/// <see cref="FusedRowsMax"/> rows with fused integer dots a unit takes a run of output rows of one row over every slot, else the slot
/// products are summed after the pass.
/// </remarks>
internal sealed unsafe class MatMulIdOp : CpuOp
{
    private const int KB = 1024;
    /// <summary>The expert id of an item that selects no expert (its output rows are zero; see <c>CompileOptions.SparseExpertSlots</c>).</summary>
    private const int NoExpert = -1;

    /// <summary>The expert id of an item a chunk of a disk-backed pass leaves to another chunk (its output rows are left as they are).</summary>
    private const int OtherChunk = -2;

    /// <summary>
    /// Up to this many rows every row is computed alone, as a single row is (the fused dequantize-dot kernel once per row, or the
    /// one-row dot instead of the two-row one), so a speculative verification step (2-4 rows) gives the logits of plain one-row decoding
    /// bit for bit.
    /// </summary>
    private const int FusedRowsMax = 4;

    /// <summary>Weight rows a unit of the integer-dot path decodes at most, and fewer when the units would not keep every thread busy.</summary>
    private const int IntDotRowsPerUnit = 32;

    /// <summary>
    /// Longest rows taken in pairs (<see cref="IntDot.FusedDot2"/>): at K = 512 a pair costs 1.40 / 1.64 ns per 32 values against 2.07 / 2.33
    /// alone (Q4_K / Q6_K, the loop and final sums paid once), at K = 2048 1.29 / 1.63 against 1.32 / 1.51 (the pair's registers spill).
    /// </summary>
    private const int PairedDotMaxK = 512;

    /// <summary>Most activation values quantized on the calling thread (<see cref="QuantizeActivations"/>).</summary>
    private const long QuantizeInlineMaxValues = 16384;

    /// <summary>Weight rows prefetched ahead of the one being decoded.</summary>
    private const int PrefetchRows = 2;
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _ids;
    private readonly TRef _y;
    /// <summary>Optional second product over the same rows and ids (an MLP's gate and up projections): weight and output, or null.</summary>
    private readonly TRef? _w2;
    private readonly TRef? _y2;
    /// <summary>With a pair: the output of <c>SiLU(y) * y2</c>, or null.</summary>
    private readonly TRef? _gated;
    /// <summary>With a single product: the slot weights [T, S] and the weighted slot sum [T, N], or null.</summary>
    private readonly TRef? _slotWeights;
    private readonly TRef? _slotSum;
    private readonly Action<int> _slotSumBody;
    private readonly Action<int> _sumSlotsBody;
    private long _slotSumRows;
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
    /// <summary>The format has a two-row fused dot (<see cref="IntDot.FusedDot2"/>): an expert's single item takes its rows in pairs.</summary>
    private readonly bool _fusedDot2;
    private readonly Action<int> _body;
    private readonly Action<int> _intDotBody;
    private readonly Action<int> _quantizeBody;
    private readonly Action<int> _gatedBody;
    // Scratch sized for the largest item count: per-expert counts/offsets, sorted x and y row pointers, active experts.
    private readonly int* _cnt;
    private readonly int* _start;
    private readonly int* _active;
    private readonly int* _cursor;
    private readonly float** _xrows;
    private readonly float** _yrows;
    private readonly float** _yrows2;
    private readonly float** _hrows;
    // Integer-dot path: the quantized activation row of every sorted item, the quantized rows and the rows per unit.
    private readonly byte** _qrows;
    private byte* _activations;
    private long _activationRows;
    /// <summary>Sorted items of the float path's gated activation pass.</summary>
    private long _gatedItems;
    private long _rowsPerUnit;
    private long _units;
    /// <summary>Units per product (the units of the second product follow those of the first).</summary>
    private long _unitsPerProduct;
    private long _per;
    private long _groups;
    /// <summary>Routed experts on disk (<see cref="DiskExpertBinding"/>), or null when the weights are resident.</summary>
    private readonly DiskExpertBinding? _disk;
    /// <summary>With <see cref="_disk"/>: per expert, the address of its weights in the expert cache while the pass holds it (first and second weight).</summary>
    private readonly byte** _diskWeights;
    private readonly byte** _diskWeights2;
    /// <summary>With <see cref="_disk"/>: the distinct experts of the pass, per expert its chunk (0 for none), and the ids a chunk runs.</summary>
    private readonly int* _passExperts;
    private readonly int* _chunkOf;
    private readonly int* _chunkIds;
    /// <summary>The ids the pass runs: <see cref="_chunkIds"/> during a chunk of a disk-backed pass, else the ids input.</summary>
    private int* _runIds;
    /// <summary>
    /// During the chunks of a disk-backed pass: the integer-dot experts of the whole pass, which size the integer-dot units as the pass
    /// without chunks would (the units' row runs decide which rows the two-row dot pairs), else 0.
    /// </summary>
    private int _passIntDotExperts;
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
    /// <param name="gated">With a second product: the output of <c>SiLU(y) * y2</c> ([T, S, N], dense), or null.</param>
    /// <param name="slotWeights">Without a second product: the weights of the slots ([T, S] values, dense), or null.</param>
    /// <param name="slotSum">With <paramref name="slotWeights"/>: the output of the weighted slot sum ([T, N], dense); <paramref name="y"/> is then scratch.</param>
    /// <param name="disk">The weights' experts on disk, read through the expert cache, or null when the weights are resident.</param>
    public MatMulIdOp(TRef x, TRef w, TRef ids, TRef y, WeightFormat fmt, bool intDot = false, TRef? w2 = null, TRef? y2 = null, TRef? gated = null,
        TRef? slotWeights = null, TRef? slotSum = null, DiskExpertBinding? disk = null)
    {
        _x = x;
        _w = w;
        _ids = ids;
        _y = y;
        _w2 = w2;
        _y2 = y2;
        _gated = w2 is null ? null : gated;
        _slotWeights = w2 is null && slotSum is not null ? slotWeights : null;
        _slotSum = _slotWeights is null ? null : slotSum;
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
        _fusedDot2 = _fusedDot && IntDot.HasFusedDot2(fmt) && _k <= PairedDotMaxK;
        long maxItems = Math.Max(1, ids.MaxElements);
        _cnt = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _start = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _active = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _cursor = (int*)Mem.Alloc(_experts * sizeof(int), false);
        _xrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _yrows = (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _yrows2 = w2 is null ? null : (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _hrows = _gated is null ? null : (float**)Mem.Alloc(maxItems * sizeof(float*), false);
        _qrows = (byte**)Mem.Alloc(maxItems * sizeof(byte*), false);
        _disk = disk is not null && (w2 is null || disk.Second is not null) ? disk : null;
        if (_disk is not null)
        {
            _diskWeights = (byte**)Mem.Alloc(_experts * sizeof(byte*), true);
            _diskWeights2 = (byte**)Mem.Alloc(_experts * sizeof(byte*), true);
            _passExperts = (int*)Mem.Alloc(_experts * sizeof(int), false);
            _chunkOf = (int*)Mem.Alloc(_experts * sizeof(int), true);
            _chunkIds = (int*)Mem.Alloc(maxItems * sizeof(int), false);
        }

        _body = Body;
        _intDotBody = IntDotBody;
        _quantizeBody = QuantizeBody;
        _gatedBody = GatedBody;
        _slotSumBody = SlotSumBody;
        _sumSlotsBody = SumSlotsBody;
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

        if (_hrows != null)
        {
            Mem.Free((byte*)_hrows);
        }
        Mem.Free((byte*)_qrows);
        if (_disk is not null)
        {
            Mem.Free((byte*)_diskWeights);
            Mem.Free((byte*)_diskWeights2);
            Mem.Free((byte*)_passExperts);
            Mem.Free((byte*)_chunkOf);
            Mem.Free((byte*)_chunkIds);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        long rows = _ids.Dim(0, p);
        long items = rows * _slots;
        if (items == 0)
        {
            return;
        }

        _runIds = (int*)_ids.P;
        if (_disk is not null)
        {
            RunFromDisk(rows, items);
            return;
        }

        RunPass(rows, items);
    }

    /// <summary>The pass over every item with the experts' weights addressable (resident, or held in the expert cache).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunPass(long rows, long items)
    {
        if (_slotSum is not null && _fusedDot && rows <= FusedRowsMax)
        {
            RunSlotSum(rows, items);
            return;
        }

        RunProducts(items);
        if (_slotSum is not null)
        {
            _slotSumRows = rows;
            CpuParallel.For((int)Math.Min(rows, CpuParallel.Threads * 2L), _sumSlotsBody);
        }
    }

    /// <summary>
    /// A pass whose experts are on disk: the distinct experts of the pass are held in the expert cache (read where missing; for the layer's
    /// first product the same experts of its later products are queued behind them), as one chunk when the cache takes them all, else in
    /// chunks of experts in id order, each chunk's items computed with the others left alone while the next chunk is read ahead. Every item's
    /// products are the same as with resident weights: an expert's items always run together.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunFromDisk(long rows, long items)
    {
        var disk = _disk!;
        int* ids = (int*)_ids.P;
        int count = 0;
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e < 0)
            {
                continue;
            }

            if ((uint)e >= (uint)_experts)
            {
                throw new InvalidOperationException($"MatMulId: expert id {e} outside [0, {_experts}).");
            }

            if (_chunkOf[e] == 0)
            {
                _chunkOf[e] = 1;
                _passExperts[count++] = e;
            }
        }

        var experts = new Span<int>(_passExperts, count);
        experts.Sort();
        int chunk = disk.Store.ChunkExperts(disk.First);
        try
        {
            if (count <= chunk)
            {
                disk.Store.Acquire(disk.First, disk.Second, experts, _diskWeights, _diskWeights2, prefetchLayer: true);
                try
                {
                    RunPass(rows, items);
                }
                finally
                {
                    disk.Store.Release(disk.First, disk.Second, experts);
                }

                return;
            }

            int chunks = (count + chunk - 1) / chunk;
            for (int c = 0; c < count; c++)
            {
                _chunkOf[_passExperts[c]] = c / chunk + 1;
            }

            new Span<int>(_cnt, _experts).Clear();
            for (long i = 0; i < items; i++)
            {
                if (ids[i] >= 0)
                {
                    _cnt[ids[i]]++;
                }
            }

            _passIntDotExperts = 0;
            foreach (int e in experts)
            {
                _passIntDotExperts += !_fusedDot || _cnt[e] <= FusedRowsMax ? 1 : 0;
            }

            for (int c = 0; c < chunks; c++)
            {
                var part = experts.Slice(c * chunk, Math.Min(chunk, count - c * chunk));
                disk.Store.Acquire(disk.First, disk.Second, part, _diskWeights, _diskWeights2, prefetchLayer: false);
                try
                {
                    if (c + 1 < chunks)
                    {
                        disk.Store.Prefetch(disk.First, disk.Second, experts.Slice((c + 1) * chunk, Math.Min(chunk, count - (c + 1) * chunk)));
                    }

                    for (long i = 0; i < items; i++)
                    {
                        int e = ids[i];
                        _chunkIds[i] = e < 0 || _chunkOf[e] == c + 1 ? e : OtherChunk;
                    }

                    _runIds = _chunkIds;
                    RunProducts(items);
                }
                finally
                {
                    _runIds = ids;
                    disk.Store.Release(disk.First, disk.Second, part);
                }
            }

            if (_slotSum is not null)
            {
                _slotSumRows = rows;
                CpuParallel.For((int)Math.Min(rows, CpuParallel.Threads * 2L), _sumSlotsBody);
            }
        }
        finally
        {
            _passIntDotExperts = 0;
            foreach (int e in experts)
            {
                _chunkOf[e] = 0;
            }
        }
    }

    /// <summary>The first byte of expert <paramref name="e"/>'s rows of the first (or <paramref name="second"/>) weight.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private byte* ExpertWeights(bool second, int e)
    {
        if (_disk is not null)
        {
            return second ? _diskWeights2[e] : _diskWeights[e];
        }

        return (second ? _w2!.P : _w.P) + e * _expertBytes;
    }

    /// <summary>The products of every item (y), sorted by expert; a pair's second product and gated activation with them.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunProducts(long items)
    {
        int* ids = _runIds;
        new Span<int>(_cnt, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e == OtherChunk)
            {
                continue;
            }

            if (e == NoExpert)
            {
                // The item selects no expert (its slot runs elsewhere): its products are zero.
                new Span<float>((float*)_y.P + i * _n, (int)_n).Clear();
                if (_y2 is { } y2)
                {
                    new Span<float>((float*)y2.P + i * _n, (int)_n).Clear();
                }

                if (_gated is { } h)
                {
                    new Span<float>((float*)h.P + i * _n, (int)_n).Clear();
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
            if (e < 0)
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

            if (_hrows != null)
            {
                _hrows[pos] = (float*)_gated!.P + i * _n;
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
        if (_gated is not null)
        {
            // The float path's products are complete only now: the gated activation of every item (the integer-dot units' items again,
            // with the same values).
            _gatedItems = off;
            CpuParallel.For((int)Math.Min(off, CpuParallel.Threads * 2L), _gatedBody);
        }
    }

    /// <summary>The gated activation of a share of the sorted items, whole rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void GatedBody(int c)
    {
        int chunks = (int)Math.Min(_gatedItems, CpuParallel.Threads * 2L);
        long per = (_gatedItems + chunks - 1) / chunks;
        for (long item = c * per; item < Math.Min(_gatedItems, (c + 1) * per); item++)
        {
            GatedActivation(_yrows[item], _yrows2![item], _hrows[item], _n);
        }
    }

    /// <summary><c>h = SiLU(gate) * up</c> over <paramref name="count"/> values, the arithmetic of the separate SiLU and Mul ops.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GatedActivation(float* gate, float* up, float* h, long count)
    {
        ScalarMath.Unary(UnaryKind.Silu, gate, h, count, 1, 0);
        for (long i = 0; i < count; i++)
        {
            h[i] *= up[i];
        }
    }

    /// <summary>Rows <paramref name="n0"/> to <paramref name="n1"/> of one expert against one item, two rows per fused dot.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void FusedRowPairs(byte* wbase, long n0, long n1, Q8Row xq, float* y, int k)
    {
        long n = n0;
        for (; n + 1 < n1; n += 2)
        {
            if (n + 2 + PrefetchRows <= n1)
            {
                Simd.Prefetch(wbase + (n + 2) * _rowBytes, PrefetchRows * _rowBytes);
            }

            IntDot.FusedDot2(_fmt, wbase + n * _rowBytes, wbase + (n + 1) * _rowBytes, xq, k, out float first, out float second);
            y[n] = first;
            y[n + 1] = second;
        }

        if (n < n1)
        {
            y[n] = IntDot.FusedDot(_fmt, wbase + n * _rowBytes, xq, k);
        }
    }

    /// <summary>
    /// The weighted slot sum of a few rows straight from the fused dots: the item rows quantized once, then units of
    /// <see cref="IntDotRowsPerUnit"/> output rows of one row, each over every slot in order (slots with id -1 add nothing).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunSlotSum(long rows, long items)
    {
        int* ids = _runIds;
        for (long i = 0; i < items; i++)
        {
            if (ids[i] != NoExpert && (uint)ids[i] >= (uint)_experts)
            {
                throw new InvalidOperationException($"MatMulId: expert id {ids[i]} outside [0, {_experts}).");
            }
        }

        long rowBytes = IntDot.RowBytes(_k);
        _activationRows = _perSlot ? items : rows;
        _activations = ThreadScratch.Get(ThreadScratch.Caller, _activationRows * rowBytes);
        QuantizeActivations(rowBytes);

        _slotSumRows = rows;
        _rowsPerUnit = IntDotRowsPerUnit;
        _groups = (_n + _rowsPerUnit - 1) / _rowsPerUnit;
        _units = rows * _groups;
        int chunks = (int)Math.Min(_units, CpuParallel.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        CpuParallel.For(chunks, _slotSumBody);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void SlotSumBody(int c)
    {
        long u0 = c * _per;
        long u1 = Math.Min(_units, u0 + _per);
        int k = (int)_k;
        long rowBytes = IntDot.RowBytes(_k);
        int* ids = _runIds;
        float* weights = (float*)_slotWeights!.P;
        float* acc = stackalloc float[IntDotRowsPerUnit];
        float* products = stackalloc float[IntDotRowsPerUnit];
        for (long unit = u0; unit < u1; unit++)
        {
            long t = unit / _groups;
            long n0 = unit % _groups * _rowsPerUnit;
            long n1 = Math.Min(_n, n0 + _rowsPerUnit);
            new Span<float>(acc, (int)(n1 - n0)).Clear();
            for (int s = 0; s < _slots; s++)
            {
                long item = t * _slots + s;
                int e = ids[item];
                if (e == NoExpert)
                {
                    continue;
                }

                float weight = weights[item];
                var xq = IntDot.RowAt(_activations + (_perSlot ? item : t) * rowBytes, _k);
                byte* wbase = ExpertWeights(false, e);
                if (_fusedDot2)
                {
                    FusedRowPairs(wbase, n0, n1, xq, products - n0, k);
                }
                else
                {
                    for (long n = n0; n < n1; n++)
                    {
                        if (n + PrefetchRows < n1)
                        {
                            Simd.Prefetch(wbase + (n + PrefetchRows) * _rowBytes, _rowBytes);
                        }

                        products[n - n0] = IntDot.FusedDot(_fmt, wbase + n * _rowBytes, xq, k);
                    }
                }

                for (long n = 0; n < n1 - n0; n++)
                {
                    acc[n] += weight * products[n];
                }
            }

            new Span<float>(acc, (int)(n1 - n0)).CopyTo(new Span<float>((float*)_slotSum!.P + t * _n + n0, (int)(n1 - n0)));
        }
    }

    /// <summary>The weighted slot sum of a share of the rows from the slot products, slots in order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void SumSlotsBody(int c)
    {
        int chunks = (int)Math.Min(_slotSumRows, CpuParallel.Threads * 2L);
        long per = (_slotSumRows + chunks - 1) / chunks;
        float* weights = (float*)_slotWeights!.P;
        for (long t = c * per; t < Math.Min(_slotSumRows, (c + 1) * per); t++)
        {
            float* sum = (float*)_slotSum!.P + t * _n;
            new Span<float>(sum, (int)_n).Clear();
            for (int s = 0; s < _slots; s++)
            {
                float weight = weights[t * _slots + s];
                float* product = (float*)_y.P + (t * _slots + s) * _n;
                long n = 0;
                // A product, then a sum (no fused multiply-add), as the separate Mul and slot sum compute it.
                for (; n + 8 <= _n; n += 8)
                {
                    (Vector256.Load(sum + n) + Vector256.Create(weight) * Vector256.Load(product + n)).Store(sum + n);
                }

                for (; n < _n; n++)
                {
                    sum[n] += weight * product[n];
                }
            }
        }
    }

    /// <summary>
    /// Orders <see cref="_active"/> so that the experts with at most <see cref="FusedRowsMax"/> items come first (in expert order, which
    /// keeps every row's result the same whatever the other experts' counts); returns the number of experts with more.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunIntDot(int* ids, long items, int activeExperts)
    {
        long rowBytes = IntDot.RowBytes(_k);
        _activationRows = _perSlot ? items : items / _slots;
        _activations = ThreadScratch.Get(ThreadScratch.Caller, _activationRows * rowBytes);
        QuantizeActivations(rowBytes);

        new Span<int>(_cursor, _experts).Clear();
        for (long i = 0; i < items; i++)
        {
            int e = ids[i];
            if (e < 0)
            {
                continue;
            }

            long pos = _start[e] + _cursor[e]++;
            _qrows[pos] = _activations + (_perSlot ? i : i / _slots) * rowBytes;
        }

        // Enough units for every thread, each a run of rows of one expert.
        // A gated pair's units compute both products of their rows.
        int unitProducts = _gated is null ? _products : 1;
        long target = CpuParallel.Threads * 4L;
        int layoutExperts = _passIntDotExperts > 0 ? _passIntDotExperts : activeExperts;
        long perExpert = Math.Max(1, (target + layoutExperts * unitProducts - 1) / (layoutExperts * unitProducts));
        _rowsPerUnit = Math.Clamp((_n + perExpert - 1) / perExpert, 1, IntDotRowsPerUnit);
        _groups = (_n + _rowsPerUnit - 1) / _rowsPerUnit;
        _unitsPerProduct = activeExperts * _groups;
        _units = _unitsPerProduct * unitProducts;
        int chunks = (int)Math.Min(_units, CpuParallel.Threads * 8L);
        _per = (_units + chunks - 1) / chunks;
        CpuParallel.For(chunks, _intDotBody);
    }

    /// <summary>
    /// Quantizes the <see cref="_activationRows"/> activation rows into <see cref="_activations"/>: on the calling thread up to
    /// <see cref="QuantizeInlineMaxValues"/> values (a decode step's ten slot rows of 512 take ~1 µs, less than a parallel dispatch), else in parallel.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void QuantizeActivations(long rowBytes)
    {
        if (_activationRows * _k <= QuantizeInlineMaxValues)
        {
            for (long r = 0; r < _activationRows; r++)
            {
                IntDot.Quantize(_x.F(r), _k, IntDot.RowAt(_activations + r * rowBytes, _k));
            }

            return;
        }

        CpuParallel.For((int)Math.Min(_activationRows, CpuParallel.Threads * 2L), _quantizeBody);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
            if (_gated is null)
            {
                IntDotRows(second, e, n0, n1, m, xr, (second ? _yrows2 : _yrows) + _start[e], k, level, u, a, b);
                continue;
            }

            float** gate = _yrows + _start[e];
            float** up = _yrows2! + _start[e];
            float** h = _hrows + _start[e];
            IntDotRows(false, e, n0, n1, m, xr, gate, k, level, u, a, b);
            IntDotRows(true, e, n0, n1, m, xr, up, k, level, u, a, b);
            for (long t = 0; t < m; t++)
            {
                GatedActivation(gate[t] + n0, up[t] + n0, h[t] + n0, n1 - n0);
            }
        }
    }

    /// <summary>Rows <paramref name="n0"/> to <paramref name="n1"/> of expert <paramref name="e"/> of the first (or <paramref name="second"/>) weight against the expert's quantized items.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void IntDotRows(bool second, int e, long n0, long n1, long m, byte** xr, float** yr, int k, CpuIsaLevel level, byte* u, float* a, float* b)
    {
        byte* wbase = ExpertWeights(second, e);
        if (_fusedDot2 && m == 1)
        {
            FusedRowPairs(wbase, n0, n1, IntDot.RowAt(xr[0], _k), yr[0], k);
            return;
        }

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

            byte* wbase = ExpertWeights(second, e);
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
