using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Causal depthwise conv with carried state (+ optional SiLU).</summary>
internal sealed unsafe class CausalConv1dOp : CpuOp
{
    readonly TRef _x, _w, _state, _y; readonly TRef? _bias; readonly bool _silu, _staticWeight;
    readonly long _c, _k;
    readonly float* _wT; // [K, C]
    readonly Action<int> _body;
    bool _transposed;
    long _t, _per;
    readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, K-1, C]
    readonly long _slotStride;
    long _u;
    public override string Name => "CausalConv1d";

    public CausalConv1dOp(TRef x, TRef w, TRef state, TRef y, bool silu, bool staticWeight, TRef? bias = null, TRef? runs = null)
    {
        _bias = bias; _x = x; _w = w; _state = state; _y = y; _silu = silu; _staticWeight = staticWeight; _runs = runs;
        _c = w.Dims[0]; _k = w.Dims[1];
        int o = runs is null ? 0 : 1;
        if (x.Dims[1] != _c || state.Dims[o] != _k - 1 || state.Dims[o + 1] != _c) throw new ArgumentException("CausalConv1d shape mismatch.");
        _slotStride = (_k - 1) * _c;
        _wT = (float*)Mem.Alloc(_c * _k * 4, false);
        _body = Body;
    }

    public override void Dispose() => Mem.Free((byte*)_wT);

    void Transpose()
    {
        float* w = (float*)_w.P;
        for (long c = 0; c < _c; c++) for (long j = 0; j < _k; j++) _wT[j * _c + c] = w[c * _w.Strides[0] + j * _w.Strides[1]];
    }

    /// <summary>Run u: (state slot, first row, row count).</summary>
    (long Slot, long Row, long Count) RunAt(long u)
    {
        if (_runs is null) return (0, 0, _t);
        int* r = (int*)_runs.P + u * 4;
        return (r[0], r[1], r[2]);
    }

    public override void Run(RunParameters p)
    {
        if (!_staticWeight || !_transposed) { Transpose(); _transposed = true; }
        _t = _x.Dim(0, p);
        if (_t == 0) return;
        _u = _runs is null ? 1 : _runs.Dim(0, p);
        long slots = _runs is null ? 1 : _state.Dims[0];
        for (long u = 0; u < _u; u++)
        {
            var (sl, r0, cnt) = RunAt(u);
            if ((ulong)sl >= (ulong)slots || r0 < 0 || cnt < 0 || r0 + cnt > _t) throw new InvalidOperationException($"CausalConv1d: bad sequence run {u}.");
            if (_runs is not null && ((int*)_runs.P)[u * 4 + 3] == 0) new Span<float>((float*)_state.P + sl * _slotStride, (int)_slotStride).Clear();
        }
        int chunks = (int)Math.Max(1, Math.Min(Par.Threads * 2L, _c / 256));
        _per = ((_c + chunks - 1) / chunks + 7) & ~7L;
        Par.For(chunks, _body);
        for (long u = 0; u < _u; u++)
        {
            var (sl, r0, cnt) = RunAt(u);
            UpdateState((float*)_state.P + sl * _slotStride, r0, cnt);
        }
    }

    /// <summary>state := last K-1 rows of [state; x[r0 .. r0+t)].</summary>
    void UpdateState(float* st, long r0, long t)
    {
        long km1 = _k - 1;
        if (km1 == 0 || t == 0) return;
        if (t >= km1)
        {
            for (long i = 0; i < km1; i++) Buffer.MemoryCopy(_x.F(r0 + t - km1 + i), st + i * _c, _c * 4, _c * 4);
        }
        else
        {
            Buffer.MemoryCopy(st + t * _c, st, (km1 - t) * _c * 4, (km1 - t) * _c * 4);
            for (long i = 0; i < t; i++) Buffer.MemoryCopy(_x.F(r0 + i), st + (km1 - t + i) * _c, _c * 4, _c * 4);
        }
    }

    void Body(int ch)
    {
        long c0 = ch * _per, c1 = Math.Min(_c, c0 + _per);
        if (c0 >= c1) return;
        long km1 = _k - 1;
        for (long u = 0; u < _u; u++)
        {
        var (sl, r0, cnt) = RunAt(u);
        float* st = (float*)_state.P + sl * _slotStride;
        for (long t = 0; t < cnt; t++)
        {
            float* y = (float*)_y.P + (r0 + t) * _c;
            for (long j = 0; j < _k; j++)
            {
                long wr = t + j;
                float* src = wr < km1 ? st + wr * _c : _x.F(r0 + wr - km1);
                float* w = _wT + j * _c;
                long i = c0;
                if (j == 0)
                {
                    if (Vector256.IsHardwareAccelerated)
                        for (; i + 8 <= c1; i += 8) (Vector256.Load(src + i) * Vector256.Load(w + i)).Store(y + i);
                    for (; i < c1; i++) y[i] = src[i] * w[i];
                }
                else
                {
                    if (Vector256.IsHardwareAccelerated)
                        for (; i + 8 <= c1; i += 8) Vector256.FusedMultiplyAdd(Vector256.Load(src + i), Vector256.Load(w + i), Vector256.Load(y + i)).Store(y + i);
                    for (; i < c1; i++) y[i] += src[i] * w[i];
                }
            }
            if (_bias is not null) { float* bi = (float*)_bias.P; for (long i = c0; i < c1; i++) y[i] += bi[i]; }
            if (_silu) Math2.Unary(UnaryKind.Silu, y + c0, y + c0, c1 - c0, 1, 0);
        }
        }
    }
}

/// <summary>Gated delta rule recurrence (per head sequential over tokens, rows of the state are independent).</summary>
internal sealed unsafe class GatedDeltaRuleOp : CpuOp
{
    readonly TRef _q, _k, _v, _g, _beta, _state, _y;
    readonly int _hk, _hv, _dk, _dv; readonly float _scale;
    readonly Action<int> _body;
    long _t; int _jChunks; int _jPer;
    readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, Hv, Dv, Dk]
    long _u;
    public override string Name => "GatedDeltaRule";

    public GatedDeltaRuleOp(TRef q, TRef k, TRef v, TRef g, TRef beta, TRef state, TRef y, float scale, TRef? runs = null)
    {
        _q = q; _k = k; _v = v; _g = g; _beta = beta; _state = state; _y = y; _scale = scale; _runs = runs;
        int o = runs is null ? 0 : 1;
        _hk = (int)q.Dims[1]; _dk = (int)q.Dims[2]; _hv = (int)v.Dims[1]; _dv = (int)v.Dims[2];
        if (k.Dims[1] != _hk || k.Dims[2] != _dk) throw new ArgumentException("GatedDeltaRule: q/k shape mismatch.");
        if (state.Dims[o] != _hv || state.Dims[o + 1] != _dv || state.Dims[o + 2] != _dk) throw new ArgumentException("GatedDeltaRule: state shape mismatch.");
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _t = _q.Dim(0, p);
        if (_t == 0) return;
        _u = _runs is null ? 1 : _runs.Dim(0, p);
        if (_runs is not null)
            for (long u = 0; u < _u; u++)
            {
                int* r = (int*)_runs.P + u * 4;
                if ((uint)r[0] >= _state.Dims[0] || r[1] < 0 || r[2] < 0 || r[1] + r[2] > _t) throw new InvalidOperationException($"GatedDeltaRule: bad sequence run {u}.");
                if (r[3] == 0) new Span<float>((float*)_state.P + (long)r[0] * _hv * _dv * _dk, _hv * _dv * _dk).Clear();
            }
        _jChunks = Math.Max(1, Math.Min(_dv / 8, (Par.Threads * 2 + _hv - 1) / _hv));
        _jPer = (_dv + _jChunks - 1) / _jChunks;
        Par.For(_hv * _jChunks, _body);
    }

    void Body(int task)
    {
        int h = task / _jChunks, jc = task % _jChunks;
        int j0 = jc * _jPer, j1 = Math.Min(_dv, j0 + _jPer);
        int hq = h % _hk;
        long gs0 = _g.Strides[0], gs1 = _g.Strides[1], bs0 = _beta.Strides[0], bs1 = _beta.Strides[1];
        for (long u = 0; u < _u; u++)
        {
        long slot = 0, t0 = 0, t1 = _t;
        if (_runs is not null) { int* r = (int*)_runs.P + u * 4; slot = r[0]; t0 = r[1]; t1 = t0 + r[2]; }
        float* S = (float*)_state.P + (slot * _hv + h) * (long)_dv * _dk;
        for (long t = t0; t < t1; t++)
        {
            float* q = _q.F(t * _hk + hq), k = _k.F(t * _hk + hq), v = _v.F(t * _hv + h);
            float decay = MathF.Exp(((float*)_g.P)[t * gs0 + h * gs1]);
            float beta = ((float*)_beta.P)[t * bs0 + h * bs1];
            float* o = (float*)_y.P + (t * _hv + h) * _dv;
            for (int j = j0; j < j1; j++)
            {
                float* row = S + (long)j * _dk;
                float kv = Simd.Dot(row, k, _dk) * decay;
                float delta = (v[j] - kv) * beta;
                o[j] = DecayUpdateDot(row, k, q, decay, delta, _dk) * _scale;
            }
        }
        }
    }

    /// <summary>row = row*decay + k*delta; returns dot(row, q).</summary>
    static float DecayUpdateDot(float* row, float* k, float* q, float decay, float delta, long n)
    {
        long i = 0; float sum = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var dv = Vector512.Create(decay); var lv = Vector512.Create(delta); var acc = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var r = Vector512.FusedMultiplyAdd(Vector512.Load(k + i), lv, Vector512.Load(row + i) * dv);
                r.Store(row + i);
                acc = Vector512.FusedMultiplyAdd(r, Vector512.Load(q + i), acc);
            }
            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var dv = Vector256.Create(decay); var lv = Vector256.Create(delta); var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var r = Vector256.FusedMultiplyAdd(Vector256.Load(k + i), lv, Vector256.Load(row + i) * dv);
                r.Store(row + i);
                acc = Vector256.FusedMultiplyAdd(r, Vector256.Load(q + i), acc);
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < n; i++) { float r = row[i] * decay + k[i] * delta; row[i] = r; sum += r * q[i]; }
        return sum;
    }
}
