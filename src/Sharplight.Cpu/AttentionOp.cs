using System.Runtime.Intrinsics;
using Sharplight.Tensors;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Scaled dot-product attention. Covers CachedAttention (K/V are state tensors) and plain Attention.</summary>
internal sealed unsafe class AttentionOp : CpuOp
{
    readonly TRef _q, _k, _v, _y; readonly TRef? _mask;
    readonly float _scale;
    readonly string? _offsetScalar;   // non-null => cached attention
    readonly bool _causal;
    readonly int _hq, _hkv, _d, _dv;
    readonly bool _kHalf, _vHalf;
    readonly Action<int> _body;
    long _t, _past, _s, _tasks, _per;
    [ThreadStatic] static float[]? s_scores;

    public override string Name => _offsetScalar is null ? "Attention" : "CachedAttention";

    public AttentionOp(TRef q, TRef k, TRef v, TRef? mask, TRef y, float scale, string? offsetScalar, bool causal)
    {
        _q = q; _k = k; _v = v; _y = y; _mask = mask; _scale = scale; _offsetScalar = offsetScalar; _causal = causal;
        _hq = (int)q.Dims[1]; _d = (int)q.Dims[2];
        _hkv = (int)k.Dims[1]; _dv = (int)v.Dims[2];
        _kHalf = k.Type == DataType.F16; _vHalf = v.Type == DataType.F16;
        if (_hq % _hkv != 0) throw new ArgumentException("Hq must be a multiple of Hkv.");
        if (k.Dims[2] != _d) throw new ArgumentException("K head size must equal Q head size.");
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _t = _q.Dim(0, p);
        if (_t == 0) return;
        _s = _k.Dim(0, p);
        if (_offsetScalar is not null)
        {
            if (!p.Scalars.TryGetValue(_offsetScalar, out _past)) throw new InvalidOperationException($"Runtime scalar '{_offsetScalar}' was not provided.");
            if (_past < 0 || _past + _t > _k.Dims[0]) throw new InvalidOperationException($"KV cache overflow: n_past {_past} + T {_t} > context {_k.Dims[0]}.");
        }
        _tasks = _t * _hq;
        int chunks = (int)Math.Min(_tasks, Par.Threads * 4L);
        _per = (_tasks + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long s = c * _per, e = Math.Min(_tasks, s + _per);
        long maxKv = _offsetScalar is not null ? _k.Dims[0] : Math.Max(_s, _k.Dims[0]);
        var scores = s_scores;
        if (scores is null || scores.Length < maxKv) s_scores = scores = new float[Math.Max(maxKv, 1024)];
        int group = _hq / _hkv;
        long ks0 = _k.Strides[0], ks1 = _k.Strides[1], vs0 = _v.Strides[0], vs1 = _v.Strides[1];
        fixed (float* sc = scores)
        {
            for (long task = s; task < e; task++)
            {
                long t = task / _hq; int h = (int)(task - t * _hq);
                int hk = h / group;
                long n = _offsetScalar is not null ? _past + t + 1
                       : _causal ? Math.Min(_s, t + _s - _t + 1) : _s;
                float* q = _q.F(task);
                float* o = (float*)_y.P + task * _dv;
                new Span<float>(o, _dv).Clear();
                if (n <= 0) continue;
                float* mrow = _mask is null ? null : (float*)_mask.P + t * _mask.Strides[0];
                long mstride = _mask is null ? 0 : _mask.Strides[1];
                float max = float.NegativeInfinity;
                for (long j = 0; j < n; j++)
                {
                    long off = j * ks0 + hk * ks1;
                    float dot = _kHalf ? Simd.DotHalf(q, (ushort*)_k.P + off, _d) : Simd.Dot(q, (float*)_k.P + off, _d);
                    dot *= _scale;
                    if (mrow != null) dot += mrow[j * mstride];
                    sc[j] = dot;
                    if (dot > max) max = dot;
                }
                float sum = 0;
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var mv = Vector256.Create(max); var acc = Vector256<float>.Zero;
                    for (; i + 8 <= n; i += 8) { var ev = Vector256.Exp(Vector256.Load(sc + i) - mv); ev.Store(sc + i); acc += ev; }
                    sum = Vector256.Sum(acc);
                }
                for (; i < n; i++) { float ev = MathF.Exp(sc[i] - max); sc[i] = ev; sum += ev; }
                for (long j = 0; j < n; j++)
                {
                    long off = j * vs0 + hk * vs1;
                    if (_vHalf) Simd.AxpyHalf(o, (ushort*)_v.P + off, sc[j], _dv);
                    else Simd.Axpy(o, (float*)_v.P + off, sc[j], _dv);
                }
                Simd.Scale(o, o, 1f / sum, _dv);
            }
        }
    }
}

/// <summary>Writes x [T, ...] into rows [offset, offset+T) of a state [ctx, ...] (f32 → state type).</summary>
internal sealed unsafe class CacheWriteOp : CpuOp
{
    readonly TRef _state, _x; readonly string _offset;
    readonly long _inner, _rowLen;
    readonly Action<int> _body;
    long _past, _rows, _per;
    public override string Name => "CacheWrite";

    public CacheWriteOp(TRef state, TRef x, string offset)
    {
        _state = state; _x = x; _offset = offset;
        _inner = 1; for (int i = 1; i < state.Rank; i++) _inner *= state.Dims[i];
        _rowLen = x.Dims[^1];
        long xi = 1; for (int i = 1; i < x.Rank; i++) xi *= x.Dims[i];
        if (xi != _inner) throw new ArgumentException("CacheWrite: x inner size does not match state inner size.");
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long t = _x.Dim(0, p);
        if (t == 0) return;
        if (!p.Scalars.TryGetValue(_offset, out _past)) throw new InvalidOperationException($"Runtime scalar '{_offset}' was not provided.");
        if (_past < 0 || _past + t > _state.Dims[0]) throw new InvalidOperationException($"KV cache overflow: n_past {_past} + T {t} > context {_state.Dims[0]}.");
        _rows = _x.Rows(p);
        int chunks = (int)Math.Min(_rows, Par.Chunks(_rows * _rowLen, 8192));
        _per = (_rows + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long s = c * _per, e = Math.Min(_rows, s + _per);
        var type = _state.Type;
        for (long r = s; r < e; r++)
        {
            float* src = _x.F(r);
            long dstElem = _past * _inner + r * _rowLen;
            if (type == DataType.F32) Buffer.MemoryCopy(src, (float*)_state.P + dstElem, _rowLen * 4, _rowLen * 4);
            else if (type == DataType.F16) { ushort* d = (ushort*)_state.P + dstElem; for (long i = 0; i < _rowLen; i++) d[i] = Simd.FloatToHalf(src[i]); }
            else if (type == DataType.BF16) { ushort* d = (ushort*)_state.P + dstElem; for (long i = 0; i < _rowLen; i++) d[i] = Simd.FloatToBFloat(src[i]); }
            else throw new NotSupportedException($"CacheWrite into {type}");
        }
    }
}
