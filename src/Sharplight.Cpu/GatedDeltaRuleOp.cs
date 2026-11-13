using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Gated delta rule recurrence (per head sequential over tokens, rows of the state are independent).</summary>
internal sealed unsafe class GatedDeltaRuleOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _g;
    private readonly TRef _beta;
    private readonly TRef _state;
    private readonly TRef _y;
    private readonly int _hk;
    private readonly int _hv;
    private readonly int _dk;
    private readonly int _dv;
    private readonly float _scale;
    private readonly Action<int> _body;
    private long _t;
    private int _jChunks;
    private int _jPer;
    private readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, Hv, Dv, Dk]
    private long _u;
    public override string Name => "GatedDeltaRule";

    public GatedDeltaRuleOp(TRef q, TRef k, TRef v, TRef g, TRef beta, TRef state, TRef y, float scale, TRef? runs = null)
    {
        _q = q;
        _k = k;
        _v = v;
        _g = g;
        _beta = beta;
        _state = state;
        _y = y;
        _scale = scale;
        _runs = runs;
        int o = runs is null ? 0 : 1;
        _hk = (int)q.Dims[1];
        _dk = (int)q.Dims[2];
        _hv = (int)v.Dims[1];
        _dv = (int)v.Dims[2];
        if (k.Dims[1] != _hk || k.Dims[2] != _dk)
        {
            throw new ArgumentException("GatedDeltaRule: q/k shape mismatch.");
        }

        if (state.Dims[o] != _hv || state.Dims[o + 1] != _dv || state.Dims[o + 2] != _dk)
        {
            throw new ArgumentException("GatedDeltaRule: state shape mismatch.");
        }

        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _t = _q.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        _u = _runs is null ? 1 : _runs.Dim(0, p);
        if (_runs is not null)
        {
            for (long u = 0; u < _u; u++)
            {
                int* r = (int*)_runs.P + u * 4;
                if ((uint)r[0] >= _state.Dims[0] || r[1] < 0 || r[2] < 0 || r[1] + r[2] > _t)
                {
                    throw new InvalidOperationException($"GatedDeltaRule: bad sequence run {u}.");
                }

                if (r[3] == 0)
                {
                    new Span<float>((float*)_state.P + (long)r[0] * _hv * _dv * _dk, _hv * _dv * _dk).Clear();
                }
            }
        }

        _jChunks = Math.Max(1, Math.Min(_dv / 8, (CpuParallel.Threads * 2 + _hv - 1) / _hv));
        _jPer = (_dv + _jChunks - 1) / _jChunks;
        CpuParallel.For(_hv * _jChunks, _body);
    }

    private void Body(int task)
    {
        int h = task / _jChunks;
        int jc = task % _jChunks;
        int j0 = jc * _jPer;
        int j1 = Math.Min(_dv, j0 + _jPer);
        int hq = h % _hk;
        long gs0 = _g.Strides[0];
        long gs1 = _g.Strides[1];
        long bs0 = _beta.Strides[0];
        long bs1 = _beta.Strides[1];
        for (long u = 0; u < _u; u++)
        {
        long slot = 0;
        long t0 = 0;
        long t1 = _t;
        if (_runs is not null)
        {
            int* r = (int*)_runs.P + u * 4;
            slot = r[0];
            t0 = r[1];
            t1 = t0 + r[2];
        }
        float* S = (float*)_state.P + (slot * _hv + h) * (long)_dv * _dk;
        for (long t = t0; t < t1; t++)
        {
            float* q = _q.F(t * _hk + hq);
            float* k = _k.F(t * _hk + hq);
            float* v = _v.F(t * _hv + h);
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
    private static float DecayUpdateDot(float* row, float* k, float* q, float decay, float delta, long n)
    {
        long i = 0;
        float sum = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var dv = Vector512.Create(decay);
            var lv = Vector512.Create(delta);
            var acc = Vector512<float>.Zero;
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
            var dv = Vector256.Create(decay);
            var lv = Vector256.Create(delta);
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var r = Vector256.FusedMultiplyAdd(Vector256.Load(k + i), lv, Vector256.Load(row + i) * dv);
                r.Store(row + i);
                acc = Vector256.FusedMultiplyAdd(r, Vector256.Load(q + i), acc);
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < n; i++)
        {
            float r = row[i] * decay + k[i] * delta;
            row[i] = r;
            sum += r * q[i];
        }
        return sum;
    }
}
