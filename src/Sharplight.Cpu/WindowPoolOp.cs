using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Softmax-gated pooling of windows of positions with a ring of the last rows as state (<c>WindowPool</c>, see
/// <see cref="Graph.StandardOps.WindowPool"/>): the row closing a window outputs the pooled window, every other row zeros; rows of earlier passes
/// come from the ring, which is updated with the last rows of each run afterwards.
/// </summary>
internal sealed unsafe class WindowPoolOp : CpuOp
{
    private readonly TRef _values;
    private readonly TRef _scores;
    private readonly TRef _bias;        // [r, C] position bias of the scores
    private readonly TRef _positions;   // [T, A] i32
    private readonly TRef _state;       // [P, 2C] or [S, P, 2C]
    private readonly TRef _y;           // [T, C] or [T, C/2] with overlap
    private readonly TRef? _rows;       // multi-sequence: [T, 4] (slot, position, run rows, run index)
    private readonly TRef? _runs;       // multi-sequence: [U, 4] (slot, first row, row count, first position)
    private readonly long _ratio;
    private readonly bool _overlap;
    private readonly long _channels;    // C
    private readonly long _outChannels;
    private readonly long _ring;        // P
    private readonly Action<int> _body;
    private long _t;
    private long _per;

    public override string Name => "WindowPool";

    public WindowPoolOp(TRef values, TRef scores, TRef bias, TRef positions, TRef state, TRef y, long ratio, bool overlap, TRef? rows = null, TRef? runs = null)
    {
        _values = values;
        _scores = scores;
        _bias = bias;
        _positions = positions;
        _state = state;
        _y = y;
        _rows = rows;
        _runs = runs;
        _ratio = ratio;
        _overlap = overlap;
        _channels = values.Dims[1];
        _outChannels = overlap ? _channels / 2 : _channels;
        _ring = overlap ? 2 * ratio : ratio;
        int o = rows is null ? 0 : 1;
        if (ratio < 1 || scores.Dims[1] != _channels || bias.Dims[0] != ratio || bias.Dims[1] != _channels || state.Dims[o] != _ring
            || state.Dims[o + 1] != 2 * _channels || (overlap && _channels % 2 != 0))
        {
            throw new ArgumentException("WindowPool shape mismatch.");
        }

        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _t = _values.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_t, CpuParallel.Threads * 2L);
        _per = (_t + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);

        // The ring keeps the last P rows of every run (the rows of an earlier pass that are still needed stay where they are).
        long runCount = _runs is null ? 1 : _runs.Dim(0, p);
        for (long u = 0; u < runCount; u++)
        {
            var (slot, firstRow, count, _) = Run(u);
            for (long row = Math.Max(firstRow, firstRow + count - _ring); row < firstRow + count; row++)
            {
                float* ring = RingRow(slot, Position(row));
                float* v = _values.F(row);
                float* s = _scores.F(row);
                float* b = Bias(Position(row));
                for (long c = 0; c < _channels; c++)
                {
                    ring[c] = v[c];
                    ring[_channels + c] = s[c] + b[c];
                }
            }
        }
    }

    // Run u: (state slot, first row, row count, first position); a single sequence is one run of all rows.
    private (long Slot, long FirstRow, long Count, long FirstPosition) Run(long u)
    {
        if (_runs is null)
        {
            return (0, 0, _t, ((int*)_positions.P)[0]);
        }

        int* r = (int*)_runs.P + u * 4;
        return (r[0], r[1], r[2], r[3]);
    }


    private long Position(long row) => _rows is null ? ((int*)_positions.P)[row * _positions.Strides[0]] : ((int*)_rows.P)[row * 4 + 1];

    private float* Bias(long position) => (float*)_bias.P + position % _ratio * _bias.Strides[0];

    private float* RingRow(long slot, long position) => (float*)_state.P + (slot * _ring + position % _ring) * 2 * _channels;

    private void Body(int chunk)
    {
        long first = chunk * _per;
        long end = Math.Min(_t, first + _per);
        for (long t = first; t < end; t++)
        {
            Pool(t);
        }
    }

    private void Pool(long t)
    {
        float* y = _y.F(t);
        long position = Position(t);
        if ((position + 1) % _ratio != 0)
        {
            new Span<float>(y, (int)_outChannels).Clear();
            return;
        }

        var (slot, firstRow, _, firstPosition) = _rows is null ? Run(0) : Run(((int*)_rows.P)[t * 4 + 3]);
        long windowStart = position + 1 - _ratio;
        long from = _overlap ? Math.Max(0, windowStart - _ratio) : windowStart;
        for (long d = 0; d < _outChannels; d++)
        {
            float max = float.NegativeInfinity;
            for (long q = from; q <= position; q++)
            {
                max = Math.Max(max, Entry(q, Channel(q, windowStart, d), slot, firstRow, firstPosition).Score);
            }

            float sum = 0f;
            float acc = 0f;
            for (long q = from; q <= position; q++)
            {
                var (value, score) = Entry(q, Channel(q, windowStart, d), slot, firstRow, firstPosition);
                float e = MathF.Exp(score - max);
                sum += e;
                acc += e * value;
            }

            y[d] = acc / sum;
        }
    }

    // The input channel output channel d reads at position q: overlapping windows take the first half of the previous window's rows and the
    // second half of their own.
    private long Channel(long q, long windowStart, long d) => _overlap && q >= windowStart ? _outChannels + d : d;

    // Value and biased score of channel c at position q: from this pass's rows, or from the ring for positions of an earlier pass.
    private (float Value, float Score) Entry(long q, long c, long slot, long firstRow, long firstPosition)
    {
        if (q >= firstPosition)
        {
            long row = firstRow + (q - firstPosition);
            return (_values.F(row)[c], _scores.F(row)[c] + Bias(q)[c]);
        }

        float* ring = RingRow(slot, q);
        return (ring[c], ring[_channels + c]);
    }
}
