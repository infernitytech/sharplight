using System.Runtime.Intrinsics;
using Sharplight.Tensors;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Scaled dot-product attention. Covers CachedAttention (K/V are state tensors) and plain Attention.</summary>
internal sealed unsafe class AttentionOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _y;
    private readonly TRef? _mask;
    private readonly TRef? _sinks;
    private readonly float _scale;
    private readonly int _window;       // > 0: sliding window (a query sees only its last _window keys)
    private readonly int _chunk;        // > 0: chunked attention (a query at p sees only keys [p - p % _chunk, p])
    private readonly float _softcap;    // > 0: scores = cap * tanh(scores / cap)
    private readonly float[]? _alibiSlopes;   // ALiBi: per query head slope of the key-minus-query position added to the scores
    private readonly string? _offsetScalar;   // non-null => cached attention
    private readonly bool _causal;
    private readonly int _hq;
    private readonly int _hkv;
    private readonly int _d;
    private readonly int _dv;
    private readonly bool _kHalf;
    private readonly bool _vHalf;
    private readonly WeightFormat? _kQuant;   // q8_0 / q4_0 caches: rows are dequantized before use
    private readonly WeightFormat? _vQuant;
    private readonly Action<int> _body;
    private readonly TRef? _rows;        // multi-sequence: [T, 4] (slot, pos, ...) per query row
    private readonly TRef? _spans;       // packed sequences: [T, 2] (first key row, count) per query row
    private readonly long _slotCtx;      // multi-sequence: rows per slot in the cache
    private readonly KvPages? _pages;    // paged multi-sequence cache
    private long _t;
    private long _past;
    private long _s;
    private long _tasks;
    private long _per;
    [ThreadStatic] private static float[]? s_scores;
    [ThreadStatic] private static float[]? s_row;

    public override string Name => _offsetScalar is null ? "Attention" : "CachedAttention";

    public AttentionOp(TRef q, TRef k, TRef v, TRef? mask, TRef y, float scale, string? offsetScalar, bool causal, int window = 0, float softcap = 0f,
        TRef? rows = null, int slots = 1, TRef? sinks = null, TRef? spans = null, int valueDims = 0, int chunk = 0, float[]? alibiSlopes = null,
        KvPages? pages = null)
    {
        _alibiSlopes = alibiSlopes;
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

        _body = Body;
    }

    /// <summary>The block format of a quantized KV cache (q8_0 / q4_0, rows of whole 32-value blocks), or null.</summary>
    private static WeightFormat? QuantizedFormat(DataType type) => Quant.FormatOf(type) is { } f && Quant.CanQuantize(f) ? f : null;

    public override void Run(RunParameters p)
    {
        _t = _q.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        _s = _k.Dim(0, p);
        if (_spans is not null)
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
        _tasks = _t * _hq;
        int chunks = (int)Math.Min(_tasks, CpuParallel.Threads * 4L);
        _per = (_tasks + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_tasks, s + _per);
        long maxKv = _offsetScalar is not null ? _k.Dims[0] : Math.Max(_s, _k.Dims[0]);
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
                if (_spans is not null)
                {
                    int* sp = (int*)_spans.P + t * 2;
                    j0 = sp[0];
                    n = sp[0] + (long)sp[1];
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
                }
                else
                {
                    n = _offsetScalar is not null ? _past + t + 1
                       : _causal ? Math.Min(_s, t + _s - _t + 1) : _s;
                }

                float* q = _q.F(task);
                float* o = (float*)_y.P + task * _dv;
                new Span<float>(o, _dv).Clear();
                if (n <= j0)
                {
                    continue;
                }

                float* mrow = _mask is null ? null : (float*)_mask.P + t * _mask.Strides[0];
                long mstride = _mask is null ? 0 : _mask.Strides[1];
                if (_window > 0)
                {
                    j0 = Math.Max(j0, n - _window);
                }

                if (_chunk > 0)
                {
                    j0 = Math.Max(j0, (n - 1) / _chunk * _chunk);
                }

                float sink = _sinks is null ? float.NegativeInfinity : ((float*)_sinks.P)[h * _sinks.Strides[0]];   // learned per-head logit that attends to nothing
                float max = sink;
                for (long j = j0; j < n; j++)
                {
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
                        dot += _alibiSlopes[h] * (j - (n - 1));   // the query sits at the last visible key
                    }

                    if (mrow != null)
                    {
                        dot += mrow[j * mstride];
                    }

                    sc[j] = dot;
                    if (dot > max)
                    {
                        max = dot;
                    }
                }
                float sum = _sinks is null ? 0f : MathF.Exp(sink - max);
                long i = j0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var mv = Vector256.Create(max);
                    var acc = Vector256<float>.Zero;
                    for (; i + 8 <= n; i += 8)
                    {
                        var ev = Vector256.Exp(Vector256.Load(sc + i) - mv);
                        ev.Store(sc + i);
                        acc += ev;
                    }
                    sum += Vector256.Sum(acc);
                }
                for (; i < n; i++)
                {
                    float ev = MathF.Exp(sc[i] - max);
                    sc[i] = ev;
                    sum += ev;
                }
                for (long j = j0; j < n; j++)
                {
                    long off = (slot >= 0 ? _pages!.Row(slot, j) : kvBase + j) * vs0 + hk * vs1;
                    if (_vQuant is { } vFormat)
                    {
                        Quant.Dequantize(vFormat, _v.P + off / 32 * Quant.BlockBytes(vFormat), _dv, rowBuf);
                        Simd.Axpy(o, rowBuf, sc[j], _dv);
                    }
                    else if (_vHalf)
                    {
                        Simd.AxpyHalf(o, (ushort*)_v.P + off, sc[j], _dv);
                    }
                    else
                    {
                        Simd.Axpy(o, (float*)_v.P + off, sc[j], _dv);
                    }
                }
                Simd.Scale(o, o, 1f / sum, _dv);
            }
        }
    }
}
