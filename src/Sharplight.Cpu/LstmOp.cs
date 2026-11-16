using System.Numerics;
using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>The recurrence of one LSTM direction over a dense f32 time-major signal (see <c>StandardOps.Lstm</c>); sequential over time.</summary>
internal sealed unsafe class LstmOp : CpuOp
{
    private readonly TRef _gates;
    private readonly TRef _w;
    private readonly TRef _y;
    private readonly bool _reverse;

    /// <summary>Creates the op.</summary>
    public LstmOp(TRef gates, TRef w, TRef y, bool reverse)
    {
        _gates = gates;
        _w = w;
        _y = y;
        _reverse = reverse;
    }

    public override string Name => "Lstm";

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        long steps = _gates.Dim(0, p);
        int h = (int)_w.Dims[1];
        float* gates = (float*)_gates.P;
        float* w = (float*)_w.P;
        float* y = (float*)_y.P;
        var c = new float[h];
        var prev = new float[h];
        var z = new float[4 * h];
        for (long s = 0; s < steps; s++)
        {
            long t = _reverse ? steps - 1 - s : s;
            float* g = gates + (t * 4 * h);
            for (int r = 0; r < 4 * h; r++)
            {
                z[r] = g[r] + Dot(w + ((long)r * h), prev, h);
            }

            float* o = y + (t * h);
            for (int j = 0; j < h; j++)
            {
                float i = Sigmoid(z[j]);
                float f = Sigmoid(z[h + j]);
                float gg = MathF.Tanh(z[(2 * h) + j]);
                float oo = Sigmoid(z[(3 * h) + j]);
                c[j] = (f * c[j]) + (i * gg);
                o[j] = oo * MathF.Tanh(c[j]);
            }

            new ReadOnlySpan<float>(o, h).CopyTo(prev);
        }
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    private static float Dot(float* a, float[] b, int n)
    {
        var acc = Vector<float>.Zero;
        int i = 0;
        int width = Vector<float>.Count;
        fixed (float* pb = b)
        {
            for (; i + width <= n; i += width)
            {
                acc += Vector.Load(a + i) * Vector.Load(pb + i);
            }

            float sum = Vector.Sum(acc);
            for (; i < n; i++)
            {
                sum += a[i] * pb[i];
            }

            return sum;
        }
    }
}
