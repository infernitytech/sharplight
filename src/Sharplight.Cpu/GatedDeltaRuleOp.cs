using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Gated delta rule recurrence (per head sequential over tokens, rows of the state are independent), with a per-head or per-key-channel
/// log decay, optionally time-invariant gates, and the additive (decay-only) rule. Rollback form: a record row holds, per value head, k (DK),
/// v (DV), g (DK per-channel, else 1) and beta; a tentative run advances a task-local copy of its state rows.
/// </summary>
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
    private readonly bool _channelDecay;    // g [T, Hv, Dk]
    private readonly bool _constantDecay;   // g [Hv]
    private readonly bool _constantBeta;    // beta [Hv]
    private readonly bool _additive;        // S = exp(g)·S + k ⊗ v·beta (no delta correction)
    private readonly int _gWidth;           // g floats per head in a record row
    private readonly Action<int> _body;
    private long _t;
    private int _jChunks;
    private int _jPer;
    private readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, Hv, Dv, Dk]
    private long _u;
    private readonly RollbackTables? _rollback;
    private float* _scratch;        // tentative runs: per task, a copy of its state rows
    private long _scratchFloats;
    public override string Name => "GatedDeltaRule";

    public GatedDeltaRuleOp(TRef q, TRef k, TRef v, TRef g, TRef beta, TRef state, TRef y, float scale, bool additive = false, TRef? runs = null,
        RollbackTables? rollback = null)
    {
        _rollback = rollback;
        _q = q;
        _k = k;
        _v = v;
        _g = g;
        _beta = beta;
        _state = state;
        _y = y;
        _scale = scale;
        _additive = additive;
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

        _channelDecay = g.Rank == 3;
        _constantDecay = g.Rank == 1;
        _constantBeta = beta.Rank == 1;
        if ((_channelDecay && (g.Dims[1] != _hv || g.Dims[2] != _dk)) || (_constantDecay && g.Dims[0] != _hv) || (g.Rank == 2 && g.Dims[1] != _hv)
            || (_constantBeta ? beta.Dims[0] != _hv : beta.Rank != 2 || beta.Dims[1] != _hv))
        {
            throw new ArgumentException("GatedDeltaRule: g/beta shape mismatch.");
        }

        _gWidth = _channelDecay ? _dk : 1;
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
        if (_rollback is not null)
        {
            EnsureScratch((long)_hv * _jChunks * _jPer * _dk);
        }

        CpuParallel.For(_hv * _jChunks, _body);
    }

    private void EnsureScratch(long floats)
    {
        if (_scratchFloats >= floats)
        {
            return;
        }

        Mem.Free((byte*)_scratch);
        _scratch = (float*)Mem.Alloc(floats * sizeof(float), false);
        _scratchFloats = floats;
    }

    public override void Dispose() => Mem.Free((byte*)_scratch);

    /// <summary>The log decay of token t of head h: a pointer to DK values (per-channel) or to one value.</summary>
    private float* LogDecay(long t, int h)
        => _constantDecay ? (float*)_g.P + h * _g.Strides[0]
         : _channelDecay ? (float*)_g.P + t * _g.Strides[0] + h * _g.Strides[1]
         : (float*)_g.P + t * _g.Strides[0] + h * _g.Strides[1];

    private float Beta(long t, int h) => _constantBeta ? ((float*)_beta.P)[h * _beta.Strides[0]] : ((float*)_beta.P)[t * _beta.Strides[0] + h * _beta.Strides[1]];

    private void Body(int task)
    {
        int h = task / _jChunks;
        int jc = task % _jChunks;
        int j0 = jc * _jPer;
        int j1 = Math.Min(_dv, j0 + _jPer);
        int hq = h % _hk;
        long recordHead = (long)h * (_dk + _dv + _gWidth + 1);
        float* decays = stackalloc float[_channelDecay ? _dk : 1];
        for (long u = 0; u < _u; u++)
        {
            long slot = 0;
            long t0 = 0;
            long t1 = _t;
            long position = 0;
            if (_runs is not null)
            {
                int* r = (int*)_runs.P + u * 4;
                slot = r[0];
                t0 = r[1];
                t1 = t0 + r[2];
                position = r[3];
            }

            float* S = (float*)_state.P + (slot * _hv + h) * (long)_dv * _dk;
            var plan = _rollback?.Plan(u, slot, t1 - t0, position) ?? default;
            for (int i = 0; i < plan.Replay; i++)
            {
                float* rec = plan.ReadRecords + i * _rollback!.Width + recordHead;
                float* logDecay = rec + _dk + _dv;
                float decay = Decays(logDecay, decays);
                Rows(S, rec, null, rec + _dk, decay, decays, logDecay[_gWidth], j0, j1, null);
            }

            if (plan.Tentative)
            {
                // The slot's state stays at the run's start: advance a copy of this task's rows (indexed like the state).
                float* copy = _scratch + (long)task * _jPer * _dk;
                Buffer.MemoryCopy(S + (long)j0 * _dk, copy, (long)(j1 - j0) * _dk * 4, (long)(j1 - j0) * _dk * 4);
                S = copy - (long)j0 * _dk;
            }

            for (long t = t0; t < t1; t++)
            {
                float* q = _q.F(t * _hk + hq);
                float* k = _k.F(t * _hk + hq);
                float* v = _v.F(t * _hv + h);
                float* logDecay = LogDecay(t, h);
                float beta = Beta(t, h);
                float decay = Decays(logDecay, decays);
                if (plan.Tentative)
                {
                    Record(plan.WriteRecords + (t - t0) * _rollback!.Width + recordHead, k, v, logDecay, beta, jc == 0, j0, j1);
                }

                Rows(S, k, q, v, decay, decays, beta, j0, j1, (float*)_y.P + (t * _hv + h) * _dv);
            }
        }
    }

    /// <summary>exp of the log decay: the scalar factor, or (per-channel decay) 1 with the DK factors written to <paramref name="decays"/>.</summary>
    private float Decays(float* logDecay, float* decays)
    {
        if (!_channelDecay)
        {
            return MathF.Exp(logDecay[0]);
        }

        for (int i = 0; i < _dk; i++)
        {
            decays[i] = MathF.Exp(logDecay[i]);
        }

        return 1f;
    }

    /// <summary>One token on state rows [j0, j1): decay, update with k and v (delta or additive rule), and with q the outputs o[j].</summary>
    private void Rows(float* S, float* k, float* q, float* v, float decay, float* decays, float beta, int j0, int j1, float* o)
    {
        for (int j = j0; j < j1; j++)
        {
            float* row = S + (long)j * _dk;
            float delta;
            if (_channelDecay)
            {
                float kv = ScaleDot(row, decays, k, _dk);
                delta = _additive ? v[j] * beta : (v[j] - kv) * beta;
                decay = 1f;
            }
            else
            {
                delta = _additive ? v[j] * beta : (v[j] - Simd.Dot(row, k, _dk) * decay) * beta;
            }

            if (q is null)
            {
                DecayUpdate(row, k, decay, delta, _dk);
            }
            else
            {
                o[j] = DecayUpdateDot(row, k, q, decay, delta, _dk) * _scale;
            }
        }
    }

    /// <summary>Writes this task's part of a record row: v[j0, j1) and, for the first task of the head, k, g and beta.</summary>
    private void Record(float* rec, float* k, float* v, float* logDecay, float beta, bool head, int j0, int j1)
    {
        if (head)
        {
            Buffer.MemoryCopy(k, rec, _dk * 4, _dk * 4);
            Buffer.MemoryCopy(logDecay, rec + _dk + _dv, _gWidth * 4, _gWidth * 4);
            rec[_dk + _dv + _gWidth] = beta;
        }

        for (int j = j0; j < j1; j++)
        {
            rec[_dk + j] = v[j];
        }
    }

    /// <summary>row = row ⊙ d; returns dot(row, k).</summary>
    private static float ScaleDot(float* row, float* d, float* k, long n)
    {
        long i = 0;
        float sum = 0;
        if (CpuIsa.Use512)
        {
            var acc = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var r = Vector512.Load(row + i) * Vector512.Load(d + i);
                r.Store(row + i);
                acc = Vector512.FusedMultiplyAdd(r, Vector512.Load(k + i), acc);
            }
            sum = Vector512.Sum(acc);
        }
        else if (CpuIsa.Use256)
        {
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var r = Vector256.Load(row + i) * Vector256.Load(d + i);
                r.Store(row + i);
                acc = Vector256.FusedMultiplyAdd(r, Vector256.Load(k + i), acc);
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < n; i++)
        {
            float r = row[i] * d[i];
            row[i] = r;
            sum += r * k[i];
        }
        return sum;
    }

    /// <summary>row = row*decay + k*delta (replayed rows, no output); the same vector widths as <see cref="DecayUpdateDot"/>, so bit-identical rows.</summary>
    private static void DecayUpdate(float* row, float* k, float decay, float delta, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            var dv = Vector512.Create(decay);
            var lv = Vector512.Create(delta);
            for (; i + 16 <= n; i += 16)
            {
                Vector512.FusedMultiplyAdd(Vector512.Load(k + i), lv, Vector512.Load(row + i) * dv).Store(row + i);
            }
        }
        else if (CpuIsa.Use256)
        {
            var dv = Vector256.Create(decay);
            var lv = Vector256.Create(delta);
            for (; i + 8 <= n; i += 8)
            {
                Vector256.FusedMultiplyAdd(Vector256.Load(k + i), lv, Vector256.Load(row + i) * dv).Store(row + i);
            }
        }
        for (; i < n; i++)
        {
            row[i] = row[i] * decay + k[i] * delta;
        }
    }

    /// <summary>row = row*decay + k*delta; returns dot(row, q).</summary>
    private static float DecayUpdateDot(float* row, float* k, float* q, float decay, float delta, long n)
    {
        long i = 0;
        float sum = 0;
        if (CpuIsa.Use512)
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
        else if (CpuIsa.Use256)
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
