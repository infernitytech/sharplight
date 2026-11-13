using System.Numerics;

namespace Sharplight.Cpu;

/// <summary>RWKV-7 recurrence (see <c>StandardOps.Wkv7</c>); the update of a state row is vectorized over the key channels.</summary>
internal sealed unsafe class Wkv7Op : HeadRecurrenceOp
{
    private readonly TRef _r;
    private readonly TRef _w;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _a;
    private readonly TRef _b;
    private readonly TRef _y;

    public override string Name => "Wkv7";

    public Wkv7Op(TRef r, TRef w, TRef k, TRef v, TRef a, TRef b, TRef state, TRef y, TRef? runs, RollbackTables? rollback = null)
        : base(r, state, runs, rollback)
    {
        _r = r;
        _w = w;
        _k = k;
        _v = v;
        _a = a;
        _b = b;
        _y = y;
    }

    protected override void RunRows(float* state, long t0, long t1, int head, int row0, int row1)
    {
        int n = Size;
        int heads = (int)_r.Dims[1];
        for (long t = t0; t < t1; t++)
        {
            float* r = Row(_r, t, head);
            float* w = Row(_w, t, head);
            float* k = Row(_k, t, head);
            float* v = Row(_v, t, head);
            float* a = Row(_a, t, head);
            float* b = Row(_b, t, head);
            float* y = (float*)_y.P + (t * heads + head) * n;
            for (int i = row0; i < row1; i++)
            {
                float* row = state + (long)i * n;
                float sa = Simd.Dot(a, row, n);
                float vi = v[i];
                y[i] = UpdateDot(row, w, k, b, r, vi, sa, n);
            }
        }
    }

    // Record row per head: w, k, v, a, b (N each).
    protected override void ReplayRow(float* state, float* record, int head, int row0, int row1)
    {
        int n = Size;
        float* w = record + (long)head * 5 * n;
        float* k = w + n;
        float* v = k + n;
        float* a = v + n;
        float* b = a + n;
        for (int i = row0; i < row1; i++)
        {
            float* row = state + (long)i * n;
            UpdateDot(row, w, k, b, k, v[i], Simd.Dot(a, row, n), n);
        }
    }

    protected override void RecordRow(float* record, long t, int head, int row0, int row1, bool first)
    {
        int n = Size;
        float* rec = record + (long)head * 5 * n;
        if (first)
        {
            Buffer.MemoryCopy(Row(_w, t, head), rec, n * 4L, n * 4L);
            Buffer.MemoryCopy(Row(_k, t, head), rec + n, n * 4L, n * 4L);
            Buffer.MemoryCopy(Row(_a, t, head), rec + 3 * n, n * 4L, n * 4L);
            Buffer.MemoryCopy(Row(_b, t, head), rec + 4 * n, n * 4L, n * 4L);
        }

        float* v = Row(_v, t, head);
        for (int i = row0; i < row1; i++)
        {
            rec[2 * n + i] = v[i];
        }
    }

    /// <summary>row = row·w + vi·k + sa·b; returns dot(row, r).</summary>
    private static float UpdateDot(float* row, float* w, float* k, float* b, float* r, float vi, float sa, long n)
    {
        long j = 0;
        int width = Vector<float>.Count;
        var acc = Vector<float>.Zero;
        for (; j + width <= n; j += width)
        {
            var updated = Vector.Load(row + j) * Vector.Load(w + j) + Vector.Load(k + j) * vi + Vector.Load(b + j) * sa;
            updated.Store(row + j);
            acc += updated * Vector.Load(r + j);
        }

        float sum = Vector.Sum(acc);
        for (; j < n; j++)
        {
            float updated = row[j] * w[j] + k[j] * vi + b[j] * sa;
            row[j] = updated;
            sum += updated * r[j];
        }

        return sum;
    }
}
