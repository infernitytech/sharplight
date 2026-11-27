using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Base of the per-head recurrences over square state matrices (RWKV-6/7): the rows of a head's state S[i, :] are independent, so (head, row-chunk) tasks
/// run in parallel; sequence runs (multi-sequence form) select the slot's state and a run starting at position 0 clears it first.
/// Rollback form: recorded rows are replayed before a run (<see cref="ReplayRow"/>); a tentative run records its rows
/// (<see cref="RecordRow"/>) and advances a task-local copy of its state rows. A 16-bit (bf16 / f16) state is widened into a
/// task-local f32 copy, advanced one token at a time and rounded to the state type after every token (as the Vulkan kernels do),
/// then narrowed back (<see cref="RecurrentState16"/>).
/// </summary>
internal abstract unsafe class HeadRecurrenceOp : CpuOp
{
    private readonly TRef _tokens;
    private readonly TRef _state;
    private readonly TRef? _runs;
    private readonly int _heads;
    private readonly int _size;
    private readonly Action<int> _body;
    private long _t;
    private long _u;
    private int _rowChunks;
    private int _rowsPer;
    private readonly RollbackTables? _rollback;
    private readonly StateSnapshotTable? _snapshots;
    private float* _scratch;        // tentative runs and 16-bit states: per task, an f32 copy of its state rows
    private long _scratchFloats;
    private readonly int _stateBytes;   // bytes per state element: 4 (f32) or 2 (bf16, f16)
    private readonly bool _bf16;

    /// <summary>Creates the base for state [(S,) H, N, N] and a token-major input of rows [T, H, N].</summary>
    protected HeadRecurrenceOp(TRef tokens, TRef state, TRef? runs, RollbackTables? rollback, StateSnapshotTable? snapshots = null)
    {
        _rollback = rollback;
        _snapshots = snapshots;
        _tokens = tokens;
        _state = state;
        _runs = runs;
        int o = runs is null ? 0 : 1;
        _heads = (int)tokens.Dims[1];
        _size = (int)tokens.Dims[2];
        if (state.Dims[o] != _heads || state.Dims[o + 1] != _size || state.Dims[o + 2] != _size)
        {
            throw new ArgumentException($"{Name}: state shape mismatch.");
        }

        _stateBytes = RecurrentState16.ElementBytes(state.Type, Name);
        _bf16 = state.Type == Tensors.DataType.BF16;
        _body = Body;
    }

    /// <summary>Head size N.</summary>
    protected int Size => _size;

    /// <summary>Start of head <paramref name="head"/>'s row of token <paramref name="t"/> in a token-major input [T, H, N] (honors the input's strides).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static float* Row(TRef x, long t, int head) => (float*)x.P + t * x.Strides[0] + head * x.Strides[1];

    /// <summary>Advances rows [row0, row1) of head <paramref name="head"/>'s state matrix <paramref name="state"/> (N×N, row major) over tokens [t0, t1).</summary>
    protected abstract void RunRows(float* state, long t0, long t1, int head, int row0, int row1);

    /// <summary>Applies one record row (<paramref name="record"/>, the whole row) to rows [row0, row1) of the head's state, without outputs.</summary>
    protected abstract void ReplayRow(float* state, float* record, int head, int row0, int row1);

    /// <summary>
    /// Writes token <paramref name="t"/>'s part of a record row: the per-row inputs of rows [row0, row1) and, when
    /// <paramref name="first"/> (the head's first task), the inputs shared by all rows.
    /// </summary>
    protected abstract void RecordRow(float* record, long t, int head, int row0, int row1, bool first);

    /// <inheritdoc/>
    public override void Dispose() => Mem.Free((byte*)_scratch);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        _t = _tokens.Dim(0, p);
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
                    throw new InvalidOperationException($"{Name}: bad sequence run {u}.");
                }

                if (r[3] == 0)
                {
                    long slotBytes = (long)_heads * _size * _size * _stateBytes;
                    new Span<byte>(_state.P + r[0] * slotBytes, checked((int)slotBytes)).Clear();
                }
            }
        }

        _rowChunks = Math.Max(1, Math.Min(_size / 8, (CpuParallel.Threads * 2 + _heads - 1) / _heads));
        _rowsPer = (_size + _rowChunks - 1) / _rowChunks;
        if ((_rollback is not null || _stateBytes != 4) && _scratchFloats < (long)_heads * _rowChunks * _rowsPer * _size)
        {
            Mem.Free((byte*)_scratch);
            _scratchFloats = (long)_heads * _rowChunks * _rowsPer * _size;
            _scratch = (float*)Mem.Alloc(_scratchFloats * sizeof(float), false);
        }

        CpuParallel.For(_heads * _rowChunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        int head = task / _rowChunks;
        int row0 = (task % _rowChunks) * _rowsPer;
        int row1 = Math.Min(_size, row0 + _rowsPer);
        if (row0 >= row1)
        {
            return;
        }

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

            float* state = (float*)_state.P + (slot * _heads + head) * _size * _size;
            var plan = _rollback?.Plan(u, slot, t1 - t0, position) ?? default;
            long rowFloats = (long)(row1 - row0) * _size;
            ushort* stored = null;
            if (_stateBytes != 4)
            {
                // 16-bit state: work on an f32 copy of this task's rows (indexed like the state), rounded after every token.
                stored = (ushort*)_state.P + ((slot * _heads + head) * _size + row0) * (long)_size;
                state = _scratch + (long)task * _rowsPer * _size - (long)row0 * _size;
                RecurrentState16.Widen(stored, state + (long)row0 * _size, rowFloats, _bf16);
            }

            for (int i = 0; i < plan.Replay; i++)
            {
                ReplayRow(state, plan.ReadRecords + i * _rollback!.Width, head, row0, row1);
                RoundRows(state, row0, rowFloats);
            }

            if (plan.Tentative)
            {
                if (stored is not null)
                {
                    // The slot's state stays as it is after the replay; the run advances the f32 copy.
                    RecurrentState16.Narrow(state + (long)row0 * _size, stored, rowFloats, _bf16);
                }
                else
                {
                    // The slot's state stays at the run's start: advance a copy of this task's rows (indexed like the state).
                    float* copy = _scratch + (long)task * _rowsPer * _size;
                    Buffer.MemoryCopy(state + (long)row0 * _size, copy, rowFloats * 4, rowFloats * 4);
                    state = copy - (long)row0 * _size;
                }

                for (long t = t0; t < t1; t++)
                {
                    RecordRow(plan.WriteRecords + (t - t0) * _rollback!.Width, t, head, row0, row1, row0 == 0);
                }
            }

            var (k, snapshotRow) = plan.Tentative || _snapshots is null ? default : _snapshots.At(u, t1 - t0, _state.Dims[0], Name);
            if (k > 0)
            {
                // The state after the run's first k rows: this task's rows into the snapshot row.
                Advance(state, t0, t0 + k, head, row0, row1);
                long at = ((snapshotRow * _heads + head) * _size + row0) * (long)_size;
                if (stored is not null)
                {
                    RecurrentState16.Narrow(state + (long)row0 * _size, (ushort*)_state.P + at, rowFloats, _bf16);
                }
                else
                {
                    Buffer.MemoryCopy(state + (long)row0 * _size, (float*)_state.P + at, rowFloats * 4, rowFloats * 4);
                }

                t0 += k;
            }

            Advance(state, t0, t1, head, row0, row1);
            if (stored is not null && !plan.Tentative)
            {
                RecurrentState16.Narrow(state + (long)row0 * _size, stored, rowFloats, _bf16);
            }
        }
    }

    /// <summary>
    /// <see cref="RunRows"/> over tokens [t0, t1); with a 16-bit state one token at a time, rounding the rows after each.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Advance(float* state, long t0, long t1, int head, int row0, int row1)
    {
        if (_stateBytes == 4)
        {
            RunRows(state, t0, t1, head, row0, row1);
            return;
        }

        long rowFloats = (long)(row1 - row0) * _size;
        for (long t = t0; t < t1; t++)
        {
            RunRows(state, t, t + 1, head, row0, row1);
            RoundRows(state, row0, rowFloats);
        }
    }

    /// <summary>Rounds this task's rows of a 16-bit state's f32 copy (indexed like the state) to the state type; no-op for f32.</summary>
    private void RoundRows(float* state, int row0, long rowFloats)
    {
        if (_stateBytes != 4)
        {
            RecurrentState16.Round(state + (long)row0 * _size, rowFloats, _bf16);
        }
    }
}
