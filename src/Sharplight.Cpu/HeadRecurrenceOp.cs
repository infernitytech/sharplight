using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Base of the per-head recurrences over square state matrices (RWKV-6/7): the rows of a head's state S[i, :] are independent, so (head, row-chunk) tasks
/// run in parallel; sequence runs (multi-sequence form) select the slot's state and a run starting at position 0 clears it first.
/// Rollback form: recorded rows are replayed before a run (<see cref="ReplayRow"/>); a tentative run records its rows
/// (<see cref="RecordRow"/>) and advances a task-local copy of its state rows.
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
    private float* _scratch;        // tentative runs: per task, a copy of its state rows
    private long _scratchFloats;

    /// <summary>Creates the base for state [(S,) H, N, N] and a token-major input of rows [T, H, N].</summary>
    protected HeadRecurrenceOp(TRef tokens, TRef state, TRef? runs, RollbackTables? rollback)
    {
        _rollback = rollback;
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
                    new Span<float>((float*)_state.P + (long)r[0] * _heads * _size * _size, _heads * _size * _size).Clear();
                }
            }
        }

        _rowChunks = Math.Max(1, Math.Min(_size / 8, (CpuParallel.Threads * 2 + _heads - 1) / _heads));
        _rowsPer = (_size + _rowChunks - 1) / _rowChunks;
        if (_rollback is not null && _scratchFloats < (long)_heads * _rowChunks * _rowsPer * _size)
        {
            Mem.Free((byte*)_scratch);
            _scratchFloats = (long)_heads * _rowChunks * _rowsPer * _size;
            _scratch = (float*)Mem.Alloc(_scratchFloats * sizeof(float), false);
        }

        CpuParallel.For(_heads * _rowChunks, _body);
    }

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
            for (int i = 0; i < plan.Replay; i++)
            {
                ReplayRow(state, plan.ReadRecords + i * _rollback!.Width, head, row0, row1);
            }

            if (plan.Tentative)
            {
                // The slot's state stays at the run's start: advance a copy of this task's rows (indexed like the state).
                float* copy = _scratch + (long)task * _rowsPer * _size;
                Buffer.MemoryCopy(state + (long)row0 * _size, copy, (long)(row1 - row0) * _size * 4, (long)(row1 - row0) * _size * 4);
                state = copy - (long)row0 * _size;
                for (long t = t0; t < t1; t++)
                {
                    RecordRow(plan.WriteRecords + (t - t0) * _rollback!.Width, t, head, row0, row1, row0 == 0);
                }
            }

            RunRows(state, t0, t1, head, row0, row1);
        }
    }
}
