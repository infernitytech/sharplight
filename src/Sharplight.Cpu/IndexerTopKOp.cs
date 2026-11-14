using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Lightning-indexer key selection (<c>IndexerTopK</c>, see <see cref="StandardOps.IndexerTopK"/>): per query row (and per selection head with
/// <c>per_head</c>), scores aligned groups of <c>group</c> visible key positions and lists the positions of the k / group best groups (above the
/// threshold ascending, then the ties at it ascending; −1 for positions the row cannot see and unused entries), then the optional tail.
/// </summary>
internal sealed unsafe class IndexerTopKOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _keys;
    private readonly TRef? _weights;     // [T, H] head weights (null: per-head selection)
    private readonly TRef? _positionGate; // pooled groups: [G, D] gate offset per member
    private readonly TRef _y;
    private readonly IndexerSelection _selection;
    private readonly int _heads;
    private readonly int _dims;
    private readonly int _width;          // output entries per selection row
    private readonly int _selectionHeads; // rows of selections per query row
    private readonly bool _keysHalf;
    private readonly string _offsetScalar;
    private readonly TRef? _rows;        // multi-sequence: [T, 4] (slot, position, ...) per query row
    private readonly long _slotContext;  // multi-sequence: cache rows per slot
    private readonly KvPages? _pages;    // paged multi-sequence cache
    private readonly Action<int> _body;
    private long _past;
    private long _per;
    private long _tasks;
    [ThreadStatic] private static float[]? s_scores;
    [ThreadStatic] private static float[]? s_sorted;
    [ThreadStatic] private static float[]? s_pooled;

    public override string Name => "IndexerTopK";

    public IndexerTopKOp(TRef q, TRef keys, TRef? weights, TRef? positionGate, TRef y, IndexerSelection selection, string offsetScalar, TRef? rows = null,
        int slots = 1, KvPages? pages = null)
    {
        _q = q;
        _keys = keys;
        _weights = weights;
        _positionGate = positionGate;
        _y = y;
        _selection = selection;
        _heads = (int)q.Dims[1];
        _dims = (int)q.Dims[2];
        _selectionHeads = selection.PerHead ? _heads : 1;
        _width = selection.Width;
        _keysHalf = keys.Type == DataType.F16;
        _offsetScalar = offsetScalar;
        _rows = rows;
        _slotContext = keys.Dims[0] / Math.Max(1, slots);
        _pages = pages;
        int rowValues = selection.Pooled ? 2 * _dims : _dims;
        if (keys.Dims[1] != 1 || keys.Dims[2] != rowValues)
        {
            throw new ArgumentException($"IndexerTopK: key cache rows [{keys.Dims[1]}, {keys.Dims[2]}] do not match one row of {rowValues} values.");
        }

        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long t = _q.Dim(0, p);
        if (t == 0)
        {
            return;
        }

        if (_rows is null)
        {
            if (!p.Scalars.TryGetValue(_offsetScalar, out _past))
            {
                throw new InvalidOperationException($"Runtime scalar '{_offsetScalar}' was not provided.");
            }

            if (_past < 0 || _past + t > _keys.Dims[0])
            {
                throw new InvalidOperationException($"Indexer key cache overflow: n_past {_past} + T {t} > context {_keys.Dims[0]}.");
            }
        }
        else
        {
            int* rw = (int*)_rows.P;
            long slots = _keys.Dims[0] / _slotContext;
            for (long row = 0; row < t; row++)
            {
                if (_pages is not null)
                {
                    for (long pos = rw[row * 4 + 1]; pos >= 0; pos -= _pages.BlockSize)
                    {
                        _pages.Check(rw[row * 4], pos, _keys.Dims[0]);
                    }
                }
                else if ((uint)rw[row * 4] >= slots || (uint)rw[row * 4 + 1] >= _slotContext)
                {
                    throw new InvalidOperationException($"Sequence row {row}: slot {rw[row * 4]} / position {rw[row * 4 + 1]} out of range.");
                }
            }
        }

        _tasks = t * _selectionHeads;
        int chunks = (int)Math.Min(_tasks, CpuParallel.Threads * 4L);
        _per = (_tasks + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long first = c * _per;
        long end = Math.Min(_tasks, first + _per);
        for (long task = first; task < end; task++)
        {
            SelectRow(task / _selectionHeads, (int)(task % _selectionHeads));
        }
    }

    private void SelectRow(long t, int head)
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

        int group = _selection.Group;
        int budget = _selection.Budget / group;
        long candidates = _selection.Pooled ? visible / group : (visible + group - 1) / group;
        int* y = (int*)_y.P + (t * _selectionHeads + head) * _width;
        new Span<int>(y, _width).Fill(-1);
        if (_selection.Tail)
        {
            for (int i = 0; i < group - 1 && candidates * group + i < visible; i++)
            {
                y[_selection.Budget + i] = (int)(candidates * group + i);
            }
        }

        if (candidates <= budget)
        {
            for (long g = 0; g < candidates; g++)
            {
                Expand(y, (int)g, (int)g, visible);
            }

            return;
        }

        var scores = s_scores;
        if (scores is null || scores.Length < candidates)
        {
            s_scores = scores = new float[Math.Max(candidates, 1024)];
            s_sorted = new float[scores.Length];
        }

        for (long g = 0; g < candidates; g++)
        {
            scores[g] = GroupScore(t, head, g, visible, rowBase, slot);
        }

        // The latest groups (the query's own and the ones before it) are always picked.
        long last = (visible - 1) / group;
        for (long g = Math.Max(0, last - _selection.Local + 1); g <= last && g < candidates; g++)
        {
            scores[g] = float.PositiveInfinity;
        }

        // The budget-th best score; groups above it come first, then the ties at it, both in ascending order.
        var sorted = s_sorted!;
        scores.AsSpan(0, (int)candidates).CopyTo(sorted);
        var span = sorted.AsSpan(0, (int)candidates);
        span.Sort();
        float threshold = span[(int)candidates - budget];
        int above = 0;
        for (long g = 0; g < candidates; g++)
        {
            if (scores[g] > threshold)
            {
                above++;
            }
        }

        int slotAbove = 0;
        int slotEqual = above;
        for (long g = 0; g < candidates && (slotAbove < above || slotEqual < budget); g++)
        {
            if (scores[g] > threshold)
            {
                Expand(y, slotAbove++, (int)g, visible);
            }
            else if (scores[g] == threshold && slotEqual < budget)
            {
                Expand(y, slotEqual++, (int)g, visible);
            }
        }
    }

    // Lists the visible positions of group g in entries [index · G, (index + 1) · G) of the selection row.
    private void Expand(int* y, int index, int g, long visible)
    {
        int group = _selection.Group;
        for (int i = 0; i < group; i++)
        {
            long position = (long)g * group + i;
            y[index * group + i] = position < visible ? (int)position : -1;
        }
    }

    // Score of group g: its pooled key's score, or the largest score of its visible positions.
    private float GroupScore(long t, int head, long g, long visible, long rowBase, int slot)
    {
        int group = _selection.Group;
        if (_selection.Pooled)
        {
            return KeyScore(t, head, PooledKey(g, rowBase, slot));
        }

        float best = float.NegativeInfinity;
        long end = Math.Min(visible, (g + 1) * group);
        for (long j = g * group; j < end; j++)
        {
            float score = KeyScore(t, head, KeyRow(j, rowBase, slot));
            if (score > best)
            {
                best = score;
            }
        }

        return best;
    }

    // Σ_h w[t, h] · relu(q[t, h] · key), or q[t, head] · key with per-head selection; key is a cache row (f32 or f16) or a pooled f32 key.
    private float KeyScore(long t, int head, KeyRef key)
    {
        if (_weights is null)
        {
            return Dot(_q.F(t * _heads + head), key);
        }

        float* w = (float*)_weights.P + t * _weights.Strides[0];
        long weightStride = _weights.Strides[1];
        float score = 0f;
        for (int h = 0; h < _heads; h++)
        {
            float dot = Dot(_q.F(t * _heads + h), key);   // F indexes the rows of the last axis
            if (dot > 0f)
            {
                score += w[h * weightStride] * dot;
            }
        }

        return score;
    }

    private float Dot(float* q, KeyRef key)
        => key.Pooled is not null ? Simd.Dot(q, key.Pooled, _dims)
            : _keysHalf ? Simd.DotHalf(q, (ushort*)_keys.P + key.Row * _keys.Strides[0], _dims)
            : Simd.Dot(q, (float*)_keys.P + key.Row * _keys.Strides[0], _dims);

    private KeyRef KeyRow(long position, long rowBase, int slot) => new(slot >= 0 ? _pages!.Row(slot, position) : rowBase + position, null);

    // The pooled key of group g: per channel d, Σ_i softmax_i(gate[i, d] + positionGate[i, d]) · key[i, d] over the group's G members.
    private KeyRef PooledKey(long g, long rowBase, int slot)
    {
        int group = _selection.Group;
        var pooled = s_pooled;
        if (pooled is null || pooled.Length < _dims)
        {
            s_pooled = pooled = GC.AllocateUninitializedArray<float>(_dims, pinned: true);   // pinned: its address is handed out
        }

        var max = new Span<float>((float*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref pooled[0]), _dims);
        float* sum = stackalloc float[_dims];
        float* acc = stackalloc float[_dims];
        max.Fill(float.NegativeInfinity);
        new Span<float>(sum, _dims).Clear();
        new Span<float>(acc, _dims).Clear();
        float* offsets = (float*)_positionGate!.P;
        for (int i = 0; i < group; i++)
        {
            long row = slot >= 0 ? _pages!.Row(slot, g * group + i) : rowBase + g * group + i;
            for (int d = 0; d < _dims; d++)
            {
                float key = CacheValue(row, d);
                float logit = CacheValue(row, _dims + d) + offsets[i * _positionGate.Strides[0] + d];
                if (logit > max[d])
                {
                    float rescale = MathF.Exp(max[d] - logit);
                    sum[d] *= rescale;
                    acc[d] *= rescale;
                    max[d] = logit;
                }

                float e = MathF.Exp(logit - max[d]);
                sum[d] += e;
                acc[d] += e * key;
            }
        }

        for (int d = 0; d < _dims; d++)
        {
            max[d] = acc[d] / sum[d];
        }

        return new KeyRef(-1, (float*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref pooled[0]));
    }

    private float CacheValue(long row, int index)
        => _keysHalf ? Simd.HalfToFloat(((ushort*)_keys.P)[row * _keys.Strides[0] + index]) : ((float*)_keys.P)[row * _keys.Strides[0] + index];

    /// <summary>A key to score: a cache row, or a pooled key in a per-thread buffer.</summary>
    private readonly struct KeyRef(long row, float* pooled)
    {
        public long Row { get; } = row;

        public float* Pooled { get; } = pooled;
    }
}
