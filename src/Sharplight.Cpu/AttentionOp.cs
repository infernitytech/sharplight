using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Scaled dot-product attention. Covers CachedAttention (K/V are state tensors) and plain Attention.</summary>
internal sealed unsafe partial class AttentionOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _y;
    private readonly TRef? _mask;
    private readonly TRef? _sinks;
    private readonly float _scale;
    private readonly int _window;       // > 0: sliding window (keys closer than _window positions to the query; causal: its last _window keys)
    private readonly int _chunk;        // > 0: chunked attention (a query at p sees only keys [p - p % _chunk, p])
    private readonly float _softcap;    // > 0: scores = cap * tanh(scores / cap)
    private readonly float[]? _alibiSlopes;   // ALiBi: per query head slope; scores get -slope * |key - query position|
    private readonly float[]? _positionBias;  // learned relative position bias: [Hq, 2R+1] by clamped key - query distance (T5)
    private readonly int _positionBiasRange;  // R
    private readonly TRef? _relativeBias;     // query-conditioned relative bias [T, Hq, E] by query - key distance in [0, E)
    private readonly int _relativeExtent;     // E
    private readonly string? _offsetScalar;   // non-null => cached attention
    private readonly bool _causal;
    private readonly int _hq;
    private readonly int _hkv;
    private readonly int _d;
    private readonly int _dv;
    private readonly bool _kHalf;
    private readonly bool _vHalf;
    private readonly WeightFormat? _kQuant;   // q8_0 / q4_0 / f8e4m3_b32 caches: rows are dequantized before use
    private readonly WeightFormat? _vQuant;
    private readonly Action<int> _body;
    private readonly TRef? _rows;        // multi-sequence: [T, 4] (slot, pos, ...) per query row
    private readonly TRef? _spans;       // packed sequences: [T, 2] (first key row, count) per query row
    private readonly long _slotCtx;      // multi-sequence: rows per slot in the cache
    private readonly KvPages? _pages;    // paged multi-sequence cache
    private readonly TRef? _selection;   // sparse attention: [T, K] or [T, Hs, K] key positions per query row (and selection head; < 0: none)
    private readonly CompressedKv? _compressed;   // compressed attention: entries of complete windows join the softmax (the selection lists entries)
    private readonly long _compressedSlotRows;    // multi-sequence: compressed cache rows per slot
    private readonly KvPages? _compressedPages;   // paged multi-sequence: the block table addressing the compressed cache
    private readonly bool _cHalf;
    private readonly long _sequenceRows;  // > 0: batched form, query rows [b·R, (b+1)·R) see key rows [b·L, (b+1)·L)
    private readonly long _sequenceKeys;  // L of the batched form
    private readonly TRef? _lookahead;    // cached attention: i32 [A], rows of the pass each query row also sees after its own (A = 1: none)
    private int* _look;                   // this run's lookahead entries, or null
    private long _t;
    private long _past;
    private long _s;
    private long _tasks;
    private long _per;
    [ThreadStatic] private static float[]? s_scores;
    [ThreadStatic] private static float[]? s_row;
    [ThreadStatic] private static long[]? s_positions;

    public override string Name => (_offsetScalar is null ? "Attention" : "CachedAttention") + (_tiledLanes > 0 ? ".tiled" : "");

    public AttentionOp(TRef q, TRef k, TRef v, TRef? mask, TRef y, float scale, string? offsetScalar, bool causal, int window = 0, float softcap = 0f,
        TRef? rows = null, int slots = 1, TRef? sinks = null, TRef? spans = null, int valueDims = 0, int chunk = 0, float[]? alibiSlopes = null,
        KvPages? pages = null, float[]? positionBias = null, int positionBiasRange = 0, TRef? selection = null, CompressedKv? compressed = null, long sequenceRows = 0, long sequenceKeys = 0,
        TRef? relativeBias = null, int relativeExtent = 0, TRef? lookahead = null)
    {
        _lookahead = lookahead;
        _relativeBias = relativeBias;
        _relativeExtent = relativeExtent;
        _sequenceRows = sequenceRows;
        _sequenceKeys = sequenceKeys;
        _selection = selection;
        _compressed = compressed;
        if (compressed is not null)
        {
            _compressedSlotRows = compressed.Cache.Dims[0] / Math.Max(1, slots);
            _compressedPages = pages?.Compressed(compressed.Compression);
            _cHalf = compressed.Cache.Type == DataType.F16;
            if ((compressed.Cache.Type != DataType.F32 && compressed.Cache.Type != DataType.F16) || compressed.Cache.Dims[1] != k.Dims[1] || compressed.Cache.Dims[2] != k.Dims[2])
            {
                throw new ArgumentException("CachedAttention: the compressed cache must be an f32 or f16 cache with the key cache's row layout.");
            }
        }

        _alibiSlopes = alibiSlopes;
        _positionBias = positionBias;
        _positionBiasRange = positionBiasRange;
        _chunk = chunk;
        _pages = pages;
        _spans = spans;
        _rows = rows;
        _slotCtx = k.Dims[0] / Math.Max(1, slots);
        _window = window;
        _softcap = softcap;
        _sinks = sinks;
        _q = q;
        _k = k;
        _v = v;
        _y = y;
        _mask = mask;
        _scale = scale;
        _offsetScalar = offsetScalar;
        _causal = causal;
        _hq = (int)q.Dims[1];
        _d = (int)q.Dims[2];
        _hkv = (int)k.Dims[1];
        _dv = valueDims > 0 ? valueDims : (int)v.Dims[2];   // v_dims: values are a prefix of the cache rows (latent attention)
        _kHalf = k.Type == DataType.F16;
        _vHalf = v.Type == DataType.F16;
        _kQuant = QuantizedFormat(k.Type);
        _vQuant = QuantizedFormat(v.Type);
        if (_hq % _hkv != 0)
        {
            throw new ArgumentException("Hq must be a multiple of Hkv.");
        }

        if (k.Dims[2] != _d)
        {
            throw new ArgumentException("K head size must equal Q head size.");
        }

        if (alibiSlopes is not null && alibiSlopes.Length != _hq)
        {
            throw new ArgumentException($"ALiBi needs one slope per query head ({_hq}), got {alibiSlopes.Length}.");
        }

        if (positionBias is not null && positionBias.Length != _hq * (2 * positionBiasRange + 1))
        {
            throw new ArgumentException($"position_bias needs {_hq} x {2 * positionBiasRange + 1} entries (heads x distances), got {positionBias.Length}.");
        }

        _body = Body;
        InitTiled();
    }

    /// <summary>
    /// Validates this run's lookahead entries (one per query row, each reaching rows of the pass only: in the multi-sequence form rows of
    /// the same slot at the following positions) and makes them visible to <see cref="Body"/>.
    /// </summary>
    private void CheckLookahead(RunParameters p)
    {
        var look = _lookahead!;
        if (look.Dim(0, p) != _t)
        {
            throw new InvalidOperationException($"CachedAttention lookahead has {look.Dim(0, p)} entries for {_t} rows (1 or one per row).");
        }

        int* la = (int*)look.P;
        int* rw = _rows is null ? null : (int*)_rows.P;
        for (long t = 0; t < _t; t++)
        {
            long ahead = la[t];
            bool sameRun = ahead >= 0 && t + ahead < _t
                           && (rw == null || (rw[(t + ahead) * 4] == rw[t * 4] && rw[(t + ahead) * 4 + 1] == rw[t * 4 + 1] + ahead));
            if (!sameRun)
            {
                throw new InvalidOperationException($"CachedAttention lookahead {ahead} of row {t} reaches past the rows of its sequence in this pass.");
            }
        }

        _look = la;
    }

    /// <summary>The block format of a quantized KV cache (q8_0 / q4_0 / f8e4m3_b32, rows of whole 32-value blocks), or null.</summary>
    private static WeightFormat? QuantizedFormat(DataType type) => Quant.FormatOf(type) is { } f && Quant.CanQuantize(f) ? f : null;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        _t = _q.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        _s = _k.Dim(0, p);
        if (_sequenceRows > 0)
        {
            if (_t % _sequenceRows != 0 || _s != _t / _sequenceRows * _sequenceKeys)
            {
                throw new InvalidOperationException($"Batched attention: {_t} query rows and {_s} key rows are not whole sequences of {_sequenceRows} / {_sequenceKeys} rows.");
            }
        }
        else if (_spans is not null)
        {
            int* sp = (int*)_spans.P;
            for (long t = 0; t < _t; t++)
            {
                if (sp[t * 2] < 0 || sp[t * 2 + 1] < 0 || sp[t * 2] + (long)sp[t * 2 + 1] > _s)
                {
                    throw new InvalidOperationException($"Attention span of row {t} ({sp[t * 2]}, {sp[t * 2 + 1]}) exceeds the {_s} key rows.");
                }
            }
        }
        else if (_rows is not null)
        {
            int* rw = (int*)_rows.P;
            long slots = _k.Dims[0] / _slotCtx;
            for (long t = 0; t < _t; t++)
            {
                if (_pages is not null)
                {
                    for (long pos = rw[t * 4 + 1]; pos >= 0; pos -= _pages.BlockSize)   // every block the row attends to
                    {
                        _pages.Check(rw[t * 4], pos, _k.Dims[0]);
                    }

                    if (_compressedPages is not null)
                    {
                        long step = Math.Max(1, _pages.BlockSize / _compressedPages.Compression);
                        for (long entry = (rw[t * 4 + 1] + 1) / _compressedPages.Compression - 1; entry >= 0; entry -= step)
                        {
                            _compressedPages.Check(rw[t * 4], entry, _compressed!.Cache.Dims[0]);
                        }
                    }
                }
                else if ((uint)rw[t * 4] >= slots || (uint)rw[t * 4 + 1] >= _slotCtx)
                {
                    throw new InvalidOperationException($"Sequence row {t}: slot {rw[t * 4]} / position {rw[t * 4 + 1]} out of range.");
                }
            }
        }
        else if (_offsetScalar is not null)
        {
            if (!p.Scalars.TryGetValue(_offsetScalar, out _past))
            {
                throw new InvalidOperationException($"Runtime scalar '{_offsetScalar}' was not provided.");
            }

            if (_past < 0 || _past + _t > _k.Dims[0])
            {
                throw new InvalidOperationException($"KV cache overflow: n_past {_past} + T {_t} > context {_k.Dims[0]}.");
            }
        }
        _look = null;
        if (_lookahead is not null && _lookahead.Dim(0, p) > 1)
        {
            CheckLookahead(p);
        }

        if (_tiledLanes > 0 && _look == null)
        {
            RunTiled();
            return;
        }

        _tasks = _t * _hq;
        int chunks = (int)Math.Min(_tasks, CpuParallel.Threads * 4L);
        _per = (_tasks + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);

    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_tasks, s + _per);
        long maxKv = (_offsetScalar is not null ? _k.Dims[0] : Math.Max(_s, _k.Dims[0])) + (_compressed?.Cache.Dims[0] ?? 0);
        var scores = s_scores;
        if (scores is null || scores.Length < maxKv)
        {
            s_scores = scores = new float[Math.Max(maxKv, 1024)];
        }

        var row = s_row;
        if (row is null || row.Length < Math.Max(_d, _dv))
        {
            s_row = row = new float[Math.Max(_d, _dv)];
        }

        int group = _hq / _hkv;
        long ks0 = _k.Strides[0];
        long ks1 = _k.Strides[1];
        long vs0 = _v.Strides[0];
        long vs1 = _v.Strides[1];
        fixed (float* sc = scores, rowBuf = row)
        {
            for (long task = s; task < e; task++)
            {
                long t = task / _hq;
                int h = (int)(task - t * _hq);
                int hk = h / group;
                long n;
                long kvBase = 0;
                int slot = -1;   // paged: the query's slot
                long j0 = 0;
                long queryPosition;   // the query's position on the key axis (window and ALiBi distances)
                if (_sequenceRows > 0)
                {
                    j0 = t / _sequenceRows * _sequenceKeys;
                    n = j0 + _sequenceKeys;
                    queryPosition = t;
                }
                else if (_spans is not null)
                {
                    int* sp = (int*)_spans.P + t * 2;
                    j0 = sp[0];
                    n = sp[0] + (long)sp[1];
                    queryPosition = t;   // packed self-attention: query row t is key row t
                    if (_causal)
                    {
                        n = Math.Min(n, t + 1);
                    }
                }
                else if (_rows is not null)
                {
                    int* rw = (int*)_rows.P + t * 4;
                    kvBase = rw[0] * _slotCtx;
                    slot = _pages is null ? -1 : rw[0];
                    n = rw[1] + 1;
                    queryPosition = rw[1];
                }
                else
                {
                    queryPosition = _offsetScalar is not null ? _past + t : t + _s - _t;
                    n = _offsetScalar is not null ? _causal ? _past + t + 1 : _past + _t
                       : _causal ? Math.Min(_s, t + _s - _t + 1) : _s;
                }

                if (_window > 0)
                {
                    // Keys closer than _window positions; for causal attention only the end before the query remains.
                    j0 = Math.Max(j0, queryPosition - _window + 1);
                    n = Math.Min(n, queryPosition + _window);
                }

                if (_look != null)
                {
                    n += _look[t];   // the later rows of the pass this row also sees (the rest of its image)
                }

                float* q = _q.F(task);
                float* o = (float*)_y.P + task * _dv;
                new Span<float>(o, _dv).Clear();
                long* positions = null;
                if (_selection is not null && _compressed is null)
                {
                    // Sparse attention: the scores are kept per selected entry, entries index the visible selected positions.
                    positions = SelectedPositions(t, h, n, out long count);
                    j0 = 0;
                    n = count;
                }

                long compressedCount = 0;
                long* entries = null;
                if (_compressed is not null)
                {
                    long complete = (queryPosition + 1) / _compressed.Compression;
                    if (_selection is not null)
                    {
                        entries = SelectedPositions(t, h, complete, out compressedCount);
                    }
                    else
                    {
                        compressedCount = complete;
                    }
                }

                if (n <= j0 && compressedCount == 0)
                {
                    continue;
                }

                n = Math.Max(n, j0);
                float* mrow = MaskRow(t);
                long mstride = _mask is null ? 0 : MaskColumnStride;
                if (_chunk > 0)
                {
                    j0 = Math.Max(j0, (n - 1) / _chunk * _chunk);
                }

                float* relativeRow = _relativeBias is null ? null : (float*)_relativeBias.P + task * _relativeExtent;
                // A learned logit that attends to nothing: one per head, or one per query row and head (sinks [T, Hq]).
                float sink = _sinks is null ? float.NegativeInfinity
                    : _sinks.Dims.Length == 2 ? ((float*)_sinks.P)[t * _sinks.Strides[0] + h * _sinks.Strides[1]] : ((float*)_sinks.P)[h * _sinks.Strides[0]];
                float max = sink;
                for (long entry = j0; entry < n; entry++)
                {
                    long j = positions == null ? entry : positions[entry];
                    long off = (slot >= 0 ? _pages!.Row(slot, j) : kvBase + j) * ks0 + hk * ks1;
                    float dot;
                    if (_kQuant is { } kFormat)
                    {
                        Quant.Dequantize(kFormat, _k.P + off / 32 * Quant.BlockBytes(kFormat), _d, rowBuf);
                        dot = Simd.Dot(q, rowBuf, _d);
                    }
                    else
                    {
                        dot = _kHalf ? Simd.DotHalf(q, (ushort*)_k.P + off, _d) : Simd.Dot(q, (float*)_k.P + off, _d);
                    }

                    dot *= _scale;
                    if (_softcap > 0)
                    {
                        dot = _softcap * MathF.Tanh(dot / _softcap);
                    }

                    if (_alibiSlopes is not null)
                    {
                        dot -= _alibiSlopes[h] * Math.Abs(j - queryPosition);
                    }

                    if (_positionBias is not null)
                    {
                        long distance = Math.Clamp(j - queryPosition, -_positionBiasRange, _positionBiasRange);
                        dot += _positionBias[h * (2 * _positionBiasRange + 1) + distance + _positionBiasRange];
                    }

                    if (relativeRow != null)
                    {
                        long distance = queryPosition - j;
                        if (distance >= 0 && distance < _relativeExtent)
                        {
                            dot += relativeRow[distance];
                        }
                    }

                    if (mrow != null)
                    {
                        dot += mrow[j * mstride];
                    }

                    sc[entry] = dot;
                    if (dot > max)
                    {
                        max = dot;
                    }
                }

                // Compressed entries follow the cache rows in the score buffer: sc[n + ci] for entry i of this row.
                for (long ci = 0; ci < compressedCount; ci++)
                {
                    long off = CompressedRow(t, entries == null ? ci : entries[ci], hk);
                    float dot = (_cHalf ? Simd.DotHalf(q, (ushort*)_compressed!.Cache.P + off, _d) : Simd.Dot(q, (float*)_compressed!.Cache.P + off, _d)) * _scale;
                    sc[n + ci] = dot;
                    if (dot > max)
                    {
                        max = dot;
                    }
                }

                for (long ci = 0; ci < compressedCount; ci++)
                {
                    float ev = MathF.Exp(sc[n + ci] - max);
                    sc[n + ci] = ev;
                }

                float sum = _sinks is null ? 0f : MathF.Exp(sink - max);
                for (long ci = 0; ci < compressedCount; ci++)
                {
                    sum += sc[n + ci];
                }

                long i = j0;
                if (CpuIsa.Use256)
                {
                    var mv = Vector256.Create(max);
                    var acc = Vector256<float>.Zero;
                    for (; i + 8 <= n; i += 8)
                    {
                        var ev = VectorExp.Exp(Vector256.Load(sc + i) - mv);
                        ev.Store(sc + i);
                        acc += ev;
                    }
                    sum += Vector256.Sum(acc);
                }
                else if (CpuIsa.Use128)
                {
                    var mv = Vector128.Create(max);
                    var acc = Vector128<float>.Zero;
                    for (; i + 4 <= n; i += 4)
                    {
                        var ev = VectorExp.Exp(Vector128.Load(sc + i) - mv);
                        ev.Store(sc + i);
                        acc += ev;
                    }
                    sum += Vector128.Sum(acc);
                }
                for (; i < n; i++)
                {
                    float ev = MathF.Exp(sc[i] - max);
                    sc[i] = ev;
                    sum += ev;
                }
                for (long entry = j0; entry < n; entry++)
                {
                    long j = positions == null ? entry : positions[entry];
                    long off = (slot >= 0 ? _pages!.Row(slot, j) : kvBase + j) * vs0 + hk * vs1;
                    if (_vQuant is { } vFormat)
                    {
                        Quant.Dequantize(vFormat, _v.P + off / 32 * Quant.BlockBytes(vFormat), _dv, rowBuf);
                        Simd.Axpy(o, rowBuf, sc[entry], _dv);
                    }
                    else if (_vHalf)
                    {
                        Simd.AxpyHalf(o, (ushort*)_v.P + off, sc[entry], _dv);
                    }
                    else
                    {
                        Simd.Axpy(o, (float*)_v.P + off, sc[entry], _dv);
                    }
                }
                for (long ci = 0; ci < compressedCount; ci++)
                {
                    long off = CompressedRow(t, entries == null ? ci : entries[ci], hk);
                    if (_cHalf)
                    {
                        Simd.AxpyHalf(o, (ushort*)_compressed!.Cache.P + off, sc[n + ci], _dv);
                    }
                    else
                    {
                        Simd.Axpy(o, (float*)_compressed!.Cache.P + off, sc[n + ci], _dv);
                    }
                }

                Simd.Scale(o, o, 1f / sum, _dv);
            }
        }
    }
    /// <summary>The additive mask row of query row <paramref name="t"/> (a one-row mask serves every query), or null without a mask.</summary>
    private float* MaskRow(long t) => _mask is null ? null : (float*)_mask.P + (_mask.Dims.Length == 1 ? 0 : t * _mask.Strides[0]);

    /// <summary>Element stride between the key columns of the mask.</summary>
    private long MaskColumnStride => _mask!.Strides[^1];

    /// <summary>
    /// The selected key positions of query row <paramref name="t"/> and query head <paramref name="h"/> that it can see (below
    /// <paramref name="visible"/>), in a per-thread buffer that stays valid until the next call on the thread. A rank-3 selection [T, Hs, K] has
    /// one row per selection head; query head h reads the row of selection head h / (Hq / Hs).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private long* SelectedPositions(long t, int h, long visible, out long count)
    {
        bool perHead = _selection!.Dims.Length == 3;
        long width = _selection.Dims[^1];
        var buffer = s_positions;
        if (buffer is null || buffer.Length < width)
        {
            s_positions = buffer = GC.AllocateUninitializedArray<long>((int)width, pinned: true);   // pinned: its address is handed out
        }

        int* row = (int*)_selection.P + t * _selection.Strides[0];
        if (perHead)
        {
            row += h / (_hq / _selection.Dims[1]) * _selection.Strides[1];
        }

        long entryStride = _selection.Strides[^1];
        count = 0;
        for (long i = 0; i < width; i++)
        {
            long j = row[i * entryStride];
            if (j >= 0 && j < visible)
            {
                buffer[count++] = j;
            }
        }

        return (long*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref buffer[0]);
    }

    /// <summary>Element offset of compressed entry <paramref name="entry"/> (key-value head <paramref name="hk"/>) of query row <paramref name="t"/>'s sequence.</summary>
    private long CompressedRow(long t, long entry, int hk)
    {
        var cache = _compressed!.Cache;
        long row = entry;
        if (_rows is not null)
        {
            int* rw = (int*)_rows.P + t * 4;
            row = _compressedPages is not null ? _compressedPages.Row(rw[0], entry) : rw[0] * _compressedSlotRows + entry;
        }

        return row * cache.Strides[0] + hk * cache.Strides[1];
    }
}
