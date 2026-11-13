using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Pools row ranges of x [T, D]: spans [B, 2] (first row, count) → y [B, D] (mean / sum / max / first / last).</summary>
internal sealed unsafe class SequencePoolOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _spans;
    private readonly TRef _y;
    private readonly int _mode;
    private readonly Action<int> _body;
    private long _t;
    public override string Name => "SequencePool";

    public SequencePoolOp(TRef x, TRef spans, TRef y, string mode)
    {
        _x = x;
        _spans = spans;
        _y = y;
        _mode = mode switch { "mean" => 0, "sum" => 1, "max" => 2, "first" => 3, "last" => 4, _ => throw new ArgumentException($"SequencePool mode '{mode}'") };
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long b = _spans.Dim(0, p);
        _t = _x.Dim(0, p);
        int* sp = (int*)_spans.P;
        for (long i = 0; i < b; i++)
        {
            if (sp[i * 2] < 0 || sp[i * 2 + 1] < 0 || sp[i * 2] + (long)sp[i * 2 + 1] > _t)
            {
                throw new InvalidOperationException($"SequencePool span {i} ({sp[i * 2]}, {sp[i * 2 + 1]}) exceeds the {_t} rows.");
            }
        }

        if (b > 0)
        {
            CpuParallel.For((int)b, _body);
        }
    }

    private void Body(int i)
    {
        int* sp = (int*)_spans.P + i * 2;
        long first = sp[0];
        long count = sp[1];
        long d = _x.Dims[^1];
        float* y = (float*)_y.P + i * d;
        new Span<float>(y, (int)d).Clear();
        if (count == 0)
        {
            return;
        }

        switch (_mode)
        {
            case 3:
                Buffer.MemoryCopy(_x.F(first), y, d * 4, d * 4);
                return;
            case 4:
                Buffer.MemoryCopy(_x.F(first + count - 1), y, d * 4, d * 4);
                return;
            case 2:
                Buffer.MemoryCopy(_x.F(first), y, d * 4, d * 4);
                for (long r = 1; r < count; r++) { float* x = _x.F(first + r); for (long k = 0; k < d; k++)
                    {
                        if (x[k] > y[k])
                        {
                            y[k] = x[k];
                        }
                    }
                }
                return;
        }
        for (long r = 0; r < count; r++)
        {
            Simd.Axpy(y, _x.F(first + r), 1f, d);
        }

        if (_mode == 0)
        {
            Simd.Scale(y, y, 1f / count, d);
        }
    }
}
