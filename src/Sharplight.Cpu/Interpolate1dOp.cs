using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Resamples a dense f32 time-major signal <c>[T, C]</c> by <c>up / down</c> along time (see <c>StandardOps.Interpolate1d</c>).</summary>
internal sealed unsafe class Interpolate1dOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _linear;
    private readonly long _up;
    private readonly long _down;
    private readonly Action<int> _body;
    private long _tin;
    private long _tout;
    private long _per;

    /// <summary>Creates the op.</summary>
    public Interpolate1dOp(TRef x, TRef y, bool linear, int up, int down)
    {
        _x = x;
        _y = y;
        _linear = linear;
        _up = up;
        _down = down;
        _body = Body;
    }

    public override string Name => "Interpolate1d";

    public override void Run(RunParameters p)
    {
        _tin = _x.Dim(0, p);
        _tout = _y.Dim(0, p);
        if (_tout == 0)
        {
            return;
        }

        var (chunks, per) = OpHelpers.Split(_tout, _x.Dims[1]);
        _per = per;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int chunk)
    {
        long first = chunk * _per;
        long last = Math.Min(_tout, first + _per);
        long channels = _x.Dims[1];
        float* x = (float*)_x.P;
        float* y = (float*)_y.P;
        for (long j = first; j < last; j++)
        {
            if (!_linear)
            {
                long i = Math.Min(_tin - 1, j * _down / _up);
                new ReadOnlySpan<float>(x + (i * channels), (int)channels).CopyTo(new Span<float>(y + (j * channels), (int)channels));
                continue;
            }

            double source = ((j + 0.5) * _down / _up) - 0.5;
            if (source < 0)
            {
                source = 0;
            }

            long i0 = Math.Min((long)Math.Floor(source), _tin - 1);
            long i1 = Math.Min(i0 + 1, _tin - 1);
            float frac = (float)(source - i0);
            for (long c = 0; c < channels; c++)
            {
                float a = x[(i0 * channels) + c];
                float b = x[(i1 * channels) + c];
                y[(j * channels) + c] = a + ((b - a) * frac);
            }
        }
    }
}
