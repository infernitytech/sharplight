using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Base of the per-head recurrences over square state matrices (RWKV-6/7): the rows of a head's state S[i, :] are independent, so (head, row-chunk) tasks
/// run in parallel; sequence runs (multi-sequence form) select the slot's state and a run starting at position 0 clears it first.
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

    /// <summary>Creates the base for state [(S,) H, N, N] and a token-major input of rows [T, H, N].</summary>
    protected HeadRecurrenceOp(TRef tokens, TRef state, TRef? runs)
    {
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

    /// <summary>Advances rows [row0, row1) of head <paramref name="head"/>'s state matrix <paramref name="state"/> (N×N, row major) over tokens [t0, t1).</summary>
    protected abstract void RunRows(float* state, long t0, long t1, int head, int row0, int row1);

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
            if (_runs is not null)
            {
                int* r = (int*)_runs.P + u * 4;
                slot = r[0];
                t0 = r[1];
                t1 = t0 + r[2];
            }

            RunRows((float*)_state.P + (slot * _heads + head) * _size * _size, t0, t1, head, row0, row1);
        }
    }
}
