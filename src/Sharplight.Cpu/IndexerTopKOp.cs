using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Lightning-indexer key selection (<c>IndexerTopK</c>): per query row, scores the visible key-cache positions by
/// Σ_h w[t, h] · relu(q[t, h] · key[j]) and writes the positions of the k best (above the k-th score ascending, then the ties at it
/// ascending; −1 fills rows that see at most k positions after listing all of them).
/// </summary>
internal sealed unsafe class IndexerTopKOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _keys;
    private readonly TRef _weights;
    private readonly TRef _y;
    private readonly int _k;
    private readonly int _heads;
    private readonly int _dims;
    private readonly bool _keysHalf;
    private readonly string _offsetScalar;
    private readonly TRef? _rows;        // multi-sequence: [T, 4] (slot, position, ...) per query row
    private readonly long _slotContext;  // multi-sequence: cache rows per slot
    private readonly KvPages? _pages;    // paged multi-sequence cache
    private readonly Action<int> _body;
    private long _past;
    private long _per;
    private long _t;
    [ThreadStatic] private static float[]? s_scores;
    [ThreadStatic] private static float[]? s_sorted;

    public override string Name => "IndexerTopK";

    public IndexerTopKOp(TRef q, TRef keys, TRef weights, TRef y, int k, string offsetScalar, TRef? rows = null, int slots = 1, KvPages? pages = null)
    {
        _q = q;
        _keys = keys;
        _weights = weights;
        _y = y;
        _k = k;
        _heads = (int)q.Dims[1];
        _dims = (int)q.Dims[2];
        _keysHalf = keys.Type == DataType.F16;
        _offsetScalar = offsetScalar;
        _rows = rows;
        _slotContext = keys.Dims[0] / Math.Max(1, slots);
        _pages = pages;
        if (keys.Dims[1] != 1 || keys.Dims[2] != _dims)
        {
            throw new ArgumentException($"IndexerTopK: key cache rows [{keys.Dims[1]}, {keys.Dims[2]}] do not match one key of {_dims} values.");
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

        if (_rows is null)
        {
            if (!p.Scalars.TryGetValue(_offsetScalar, out _past))
            {
                throw new InvalidOperationException($"Runtime scalar '{_offsetScalar}' was not provided.");
            }

            if (_past < 0 || _past + _t > _keys.Dims[0])
            {
                throw new InvalidOperationException($"Indexer key cache overflow: n_past {_past} + T {_t} > context {_keys.Dims[0]}.");
            }
        }
        else
        {
            int* rw = (int*)_rows.P;
            long slots = _keys.Dims[0] / _slotContext;
            for (long t = 0; t < _t; t++)
            {
                if (_pages is not null)
                {
                    for (long pos = rw[t * 4 + 1]; pos >= 0; pos -= _pages.BlockSize)
                    {
                        _pages.Check(rw[t * 4], pos, _keys.Dims[0]);
                    }
                }
                else if ((uint)rw[t * 4] >= slots || (uint)rw[t * 4 + 1] >= _slotContext)
                {
                    throw new InvalidOperationException($"Sequence row {t}: slot {rw[t * 4]} / position {rw[t * 4 + 1]} out of range.");
                }
            }
        }

        int chunks = (int)Math.Min(_t, CpuParallel.Threads * 4L);
        _per = (_t + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long first = c * _per;
        long end = Math.Min(_t, first + _per);
        for (long t = first; t < end; t++)
        {
            SelectRow(t);
        }
    }

    private void SelectRow(long t)
    {
        long visible;
        long rowBase = 0;
        int slot = -1;
        if (_rows is not null)
        {
            int* rw = (int*)_rows.P + t * 4;
            visible = rw[1] + 1L;
            rowBase = rw[0] * _slotContext;
            slot = _pages is null ? -1 : rw[0];
        }
        else
        {
            visible = _past + t + 1;
        }

        int* y = (int*)_y.P + t * _k;
        if (visible <= _k)
        {
            for (int i = 0; i < _k; i++)
            {
                y[i] = i < visible ? i : -1;
            }

            return;
        }

        var scores = s_scores;
        if (scores is null || scores.Length < visible)
        {
            s_scores = scores = new float[Math.Max(visible, 1024)];
            s_sorted = new float[scores.Length];
        }

        var sorted = s_sorted!;
        float* w = (float*)_weights.P + t * _weights.Strides[0];
        long weightStride = _weights.Strides[1];
        for (long j = 0; j < visible; j++)
        {
            long row = slot >= 0 ? _pages!.Row(slot, j) : rowBase + j;
            float score = 0f;
            for (int h = 0; h < _heads; h++)
            {
                float* qh = _q.F(t * _heads + h);   // F indexes the rows of the last axis
                float dot = _keysHalf
                    ? Simd.DotHalf(qh, (ushort*)_keys.P + row * _keys.Strides[0], _dims)
                    : Simd.Dot(qh, (float*)_keys.P + row * _keys.Strides[0], _dims);
                if (dot > 0f)
                {
                    score += w[h * weightStride] * dot;
                }
            }

            scores[j] = score;
        }

        // The k-th best score; positions above it come first, then the ties at it, both in ascending order.
        scores.AsSpan(0, (int)visible).CopyTo(sorted);
        var span = sorted.AsSpan(0, (int)visible);
        span.Sort();
        float threshold = span[(int)visible - _k];
        int above = 0;
        for (long j = 0; j < visible; j++)
        {
            if (scores[j] > threshold)
            {
                above++;
            }
        }

        int slotAbove = 0;
        int slotEqual = above;
        for (long j = 0; j < visible && (slotAbove < above || slotEqual < _k); j++)
        {
            if (scores[j] > threshold)
            {
                y[slotAbove++] = (int)j;
            }
            else if (scores[j] == threshold && slotEqual < _k)
            {
                y[slotEqual++] = (int)j;
            }
        }
    }
}
