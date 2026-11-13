using System.Numerics;

namespace Sharplight.Cpu;

/// <summary>RWKV-6 recurrence (see <c>StandardOps.Wkv6</c>); the update of a state row is vectorized over the key channels.</summary>
internal sealed unsafe class Wkv6Op : HeadRecurrenceOp
{
    private readonly TRef _r;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _w;
    private readonly TRef _u;
    private readonly TRef _y;

    public override string Name => "Wkv6";

    public Wkv6Op(TRef r, TRef k, TRef v, TRef w, TRef u, TRef state, TRef y, TRef? runs)
        : base(r, state, runs)
    {
        _r = r;
        _k = k;
        _v = v;
        _w = w;
        _u = u;
        _y = y;
    }

    protected override void RunRows(float* state, long t0, long t1, int head, int row0, int row1)
    {
        int n = Size;
        int heads = (int)_r.Dims[1];
        float* u = (float*)_u.P + (long)head * n;
        for (long t = t0; t < t1; t++)
        {
            float* r = Row(_r, t, head);
            float* k = Row(_k, t, head);
            float* v = Row(_v, t, head);
            float* w = Row(_w, t, head);
            float* y = (float*)_y.P + (t * heads + head) * n;
            for (int i = row0; i < row1; i++)
            {
                y[i] = UpdateDot(state + (long)i * n, r, k, w, u, v[i], n);
            }
        }
    }

    private static float* Row(TRef x, long t, int head) => (float*)x.P + t * x.Strides[0] + head * x.Strides[1];

    /// <summary>Returns Σⱼ r[j]·(u[j]·k[j]·vi + row[j]) and sets row = row·w + k·vi.</summary>
    private static float UpdateDot(float* row, float* r, float* k, float* w, float* u, float vi, long n)
    {
        long j = 0;
        var acc = Vector<float>.Zero;
        int width = Vector<float>.Count;
        for (; j + width <= n; j += width)
        {
            var kv = Vector.Load(k + j) * vi;
            var current = Vector.Load(row + j);
            acc += Vector.Load(r + j) * (Vector.Load(u + j) * kv + current);
            (current * Vector.Load(w + j) + kv).Store(row + j);
        }

        float sum = Vector.Sum(acc);
        for (; j < n; j++)
        {
            float kv = k[j] * vi;
            sum += r[j] * (u[j] * kv + row[j]);
            row[j] = row[j] * w[j] + kv;
        }

        return sum;
    }
}
