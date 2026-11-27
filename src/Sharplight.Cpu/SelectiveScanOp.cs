using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Selective scan (see <c>StandardOps.SelectiveScan</c>; Mamba-2, or Mamba-1 when A is [H, N]): per head sequential over tokens; the P rows of a head's
/// state matrix S[p, :N] are independent, so (head, row-chunk) tasks run in parallel and N is vectorized. Rollback form: a record
/// row holds x (P) and the raw dt per head, then B [G, N]; a tentative run advances a task-local copy of its state rows.
/// A 16-bit (bf16 / f16) state is widened into a task-local f32 copy, updated in f32 and rounded to the state type after every token
/// (as the Vulkan kernel does), then narrowed back (<see cref="RecurrentState16"/>).
/// </summary>
internal sealed unsafe class SelectiveScanOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _dt;
    private readonly TRef _dtBias;
    private readonly TRef _a;
    private readonly TRef _b;
    private readonly TRef _c;
    private readonly TRef _d;
    private readonly TRef _state;
    private readonly TRef _y;
    private readonly int _h;
    private readonly int _p;
    private readonly int _g;
    private readonly int _n;
    private readonly float _dtMin;
    private readonly bool _aPerState;   // A is [H, N] (Mamba-1) instead of [H]
    private readonly Action<int> _body;
    private readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, H, P, N]
    private long _t;
    private long _u;
    private int _pChunks;
    private int _pPer;
    private readonly RollbackTables? _rollback;
    private readonly StateSnapshotTable? _snapshots;
    private float* _scratch;        // tentative runs and 16-bit states: per task, an f32 copy of its state rows
    private long _scratchFloats;
    private readonly int _stateBytes;   // bytes per state element: 4 (f32) or 2 (bf16, f16)
    private readonly bool _bf16;
    public override string Name => "SelectiveScan";

    public SelectiveScanOp(TRef x, TRef dt, TRef dtBias, TRef a, TRef b, TRef c, TRef d, TRef state, TRef y, float dtMin, TRef? runs = null,
        RollbackTables? rollback = null, StateSnapshotTable? snapshots = null)
    {
        _rollback = rollback;
        _snapshots = snapshots;
        _x = x;
        _dt = dt;
        _dtBias = dtBias;
        _a = a;
        _b = b;
        _c = c;
        _d = d;
        _state = state;
        _y = y;
        _dtMin = dtMin;
        _runs = runs;
        _aPerState = a.Rank == 2;
        int o = runs is null ? 0 : 1;
        _h = (int)x.Dims[1];
        _p = (int)x.Dims[2];
        _g = (int)b.Dims[1];
        _n = (int)b.Dims[2];
        if (_h % _g != 0)
        {
            throw new ArgumentException("SelectiveScan: head count must be a multiple of the group count.");
        }

        if (c.Dims[1] != _g || c.Dims[2] != _n)
        {
            throw new ArgumentException("SelectiveScan: B/C shape mismatch.");
        }

        if (state.Dims[o] != _h || state.Dims[o + 1] != _p || state.Dims[o + 2] != _n)
        {
            throw new ArgumentException("SelectiveScan: state shape mismatch.");
        }

        if (dt.Dims[1] != _h)
        {
            throw new ArgumentException("SelectiveScan: dt shape mismatch.");
        }

        _stateBytes = RecurrentState16.ElementBytes(state.Type, Name);
        _bf16 = state.Type == DataType.BF16;
        _body = Body;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        _t = _x.Dim(0, p);
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
                    throw new InvalidOperationException($"SelectiveScan: bad sequence run {u}.");
                }

                if (r[3] == 0)
                {
                    long slotBytes = (long)_h * _p * _n * _stateBytes;
                    new Span<byte>(_state.P + r[0] * slotBytes, checked((int)slotBytes)).Clear();
                }
            }
        }

        _pChunks = Math.Max(1, Math.Min(_p / 4, (CpuParallel.Threads * 2 + _h - 1) / _h));
        _pPer = (_p + _pChunks - 1) / _pChunks;
        if ((_rollback is not null || _stateBytes != 4) && _scratchFloats < (long)_h * _pChunks * _pPer * _n)
        {
            Mem.Free((byte*)_scratch);
            _scratchFloats = (long)_h * _pChunks * _pPer * _n;
            _scratch = (float*)Mem.Alloc(_scratchFloats * sizeof(float), false);
        }

        CpuParallel.For(_h * _pChunks, _body);
    }

    public override void Dispose() => Mem.Free((byte*)_scratch);

    private static float Softplus(float v) => v > 20f ? v : (float)Math.Log(1.0 + Math.Exp(v));

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        int h = task / _pChunks;
        int pc = task % _pChunks;
        int p0 = pc * _pPer;
        int p1 = Math.Min(_p, p0 + _pPer);
        if (p0 >= p1)
        {
            return;
        }

        int g = h / (_h / _g);
        float bias = ((float*)_dtBias.P)[h];
        float A = _aPerState ? 0f : ((float*)_a.P)[h];
        float D = ((float*)_d.P)[h];
        long xAt = (long)h * (_p + 1);
        long bAt = (long)_h * (_p + 1) + (long)g * _n;
        long width = _rollback?.Width ?? 0;
        bool recordsB = pc == 0 && h % (_h / _g) == 0;
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

            float* S = (float*)_state.P + (slot * _h + h) * (long)_p * _n;
            long rowFloats = (long)(p1 - p0) * _n;
            ushort* stored = null;
            if (_stateBytes != 4)
            {
                // 16-bit state: work on an f32 copy of this task's rows (indexed like the state), rounded after every update.
                stored = (ushort*)_state.P + ((slot * _h + h) * (long)_p + p0) * _n;
                S = _scratch + (long)task * _pPer * _n - (long)p0 * _n;
                RecurrentState16.Widen(stored, S + (long)p0 * _n, rowFloats, _bf16);
            }

            var plan = _rollback?.Plan(u, slot, t1 - t0, position) ?? default;
            for (int i = 0; i < plan.Replay; i++)
            {
                float* rec = plan.ReadRecords + i * width;
                StepRow(S, rec[xAt + _p], rec + bAt, rec + bAt, rec + xAt, null, h, p0, p1, bias, A, D);
            }

            if (plan.Tentative && stored is not null)
            {
                // The slot's state stays as it is after the replay; the run advances the f32 copy.
                RecurrentState16.Narrow(S + (long)p0 * _n, stored, rowFloats, _bf16);
            }
            else if (plan.Tentative)
            {
                // The slot's state stays at the run's start: advance a copy of this task's rows (indexed like the state).
                float* copy = _scratch + (long)task * _pPer * _n;
                Buffer.MemoryCopy(S + (long)p0 * _n, copy, (long)(p1 - p0) * _n * 4, (long)(p1 - p0) * _n * 4);
                S = copy - (long)p0 * _n;
            }

            var (snapRows, snapRow) = plan.Tentative || _snapshots is null ? default : _snapshots.At(u, t1 - t0, _state.Dims[0], Name);
            for (long t = t0; t < t1; t++)
            {
                if (snapRows > 0 && t == t0 + snapRows)
                {
                    StoreRows(S, snapRow, h, p0, p1);
                }

                float dtRaw = ((float*)_dt.P)[t * _dt.Strides[0] + h * _dt.Strides[1]];
                float* B = (float*)_b.P + t * _b.Strides[0] + g * _b.Strides[1];
                float* C = (float*)_c.P + t * _c.Strides[0] + g * _c.Strides[1];
                float* x = (float*)_x.P + t * _x.Strides[0] + h * _x.Strides[1];
                if (plan.Tentative)
                {
                    float* rec = plan.WriteRecords + (t - t0) * width;
                    Buffer.MemoryCopy(x + p0, rec + xAt + p0, (long)(p1 - p0) * 4, (long)(p1 - p0) * 4);
                    if (pc == 0)
                    {
                        rec[xAt + _p] = dtRaw;
                    }

                    if (recordsB)
                    {
                        Buffer.MemoryCopy(B, rec + bAt, _n * 4L, _n * 4L);
                    }
                }

                StepRow(S, dtRaw, B, C, x, (float*)_y.P + (t * _h + h) * _p, h, p0, p1, bias, A, D);
            }

            if (snapRows > 0 && t1 == t0 + snapRows)
            {
                StoreRows(S, snapRow, h, p0, p1);
            }

            if (stored is not null && !plan.Tentative)
            {
                RecurrentState16.Narrow(S + (long)p0 * _n, stored, rowFloats, _bf16);
            }
        }
    }

    /// <summary>Stores rows [p0, p1) of head h's working state <paramref name="S"/> (indexed like the state) into state row <paramref name="slot"/>.</summary>
    private void StoreRows(float* S, long slot, int h, int p0, int p1)
    {
        long floats = (long)(p1 - p0) * _n;
        long at = ((slot * _h + h) * (long)_p + p0) * _n;
        if (_stateBytes != 4)
        {
            RecurrentState16.Narrow(S + (long)p0 * _n, (ushort*)_state.P + at, floats, _bf16);
        }
        else
        {
            Buffer.MemoryCopy(S + (long)p0 * _n, (float*)_state.P + at, floats * 4, floats * 4);
        }
    }

    /// <summary>Advances rows [p0, p1) of head <paramref name="h"/> by one token; writes y unless <paramref name="y"/> is null (replay).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void StepRow(float* S, float dtRaw, float* B, float* C, float* x, float* y, int h, int p0, int p1, float bias, float A, float D)
    {
        float dt = MathF.Max(Softplus(dtRaw + bias), _dtMin);
        float decay = _aPerState ? 0f : MathF.Exp(dt * A);
        for (int p = p0; p < p1; p++)
        {
            float xp = x[p];
            float* row = S + (long)p * _n;
            float dot = _aPerState ? DecayUpdateDotPerState(row, B, C, (float*)_a.P + (long)h * _n, dt, dt * xp, _n) : DecayUpdateDot(row, B, C, decay, dt * xp, _n);
            if (y != null)
            {
                y[p] = dot + D * xp;
            }

            if (_stateBytes != 4)
            {
                RecurrentState16.Round(row, _n, _bf16);
            }
        }
    }

    /// <summary>row[i] = row[i]·exp(dt·a[i]) + b[i]·coef (a decay per state element, Mamba-1); returns dot(row, c).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DecayUpdateDotPerState(float* row, float* b, float* c, float* a, float dt, float coef, long n)
    {
        float sum = 0;
        for (long i = 0; i < n; i++)
        {
            float r = row[i] * MathF.Exp(dt * a[i]) + b[i] * coef;
            row[i] = r;
            sum += r * c[i];
        }

        return sum;
    }

    /// <summary>row = row*decay + b*coef; returns dot(row, c).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DecayUpdateDot(float* row, float* b, float* c, float decay, float coef, long n)
    {
        long i = 0;
        float sum = 0;
        if (CpuIsa.Use512)
        {
            var dv = Vector512.Create(decay);
            var cv = Vector512.Create(coef);
            var acc = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var r = Vector512.FusedMultiplyAdd(Vector512.Load(b + i), cv, Vector512.Load(row + i) * dv);
                r.Store(row + i);
                acc = Vector512.FusedMultiplyAdd(r, Vector512.Load(c + i), acc);
            }
            sum = Vector512.Sum(acc);
        }
        else if (CpuIsa.Use256)
        {
            var dv = Vector256.Create(decay);
            var cv = Vector256.Create(coef);
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var r = Vector256.FusedMultiplyAdd(Vector256.Load(b + i), cv, Vector256.Load(row + i) * dv);
                r.Store(row + i);
                acc = Vector256.FusedMultiplyAdd(r, Vector256.Load(c + i), acc);
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < n; i++)
        {
            float r = row[i] * decay + b[i] * coef;
            row[i] = r;
            sum += r * c[i];
        }
        return sum;
    }
}
