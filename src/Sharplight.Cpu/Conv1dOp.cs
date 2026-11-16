using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 1-D convolution of a dense f32 time-major signal <c>[T, C]</c> with a weight <c>[O, C/groups, K]</c> (see <c>StandardOps.Conv1d</c>);
/// parallel over output positions, zero padding handled by clipping the taps.
/// </summary>
internal sealed unsafe class Conv1dOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef? _bias;
    private readonly TRef _y;
    private readonly int _stride;
    private readonly int _padLeft;
    private readonly int _dilation;
    private readonly int _groups;
    private readonly bool _reflect;
    private readonly Action<int> _body;
    private long _tin;
    private long _tout;
    private long _per;

    /// <summary>Creates the op.</summary>
    public Conv1dOp(TRef x, TRef w, TRef? bias, TRef y, int stride, int padLeft, int dilation, int groups, bool reflect = false)
    {
        _reflect = reflect;
        _x = x;
        _w = w;
        _bias = bias;
        _y = y;
        _stride = stride;
        _padLeft = padLeft;
        _dilation = dilation;
        _groups = groups;
        _body = Body;
    }

    public override string Name => "Conv1d";

    public override void Run(RunParameters p)
    {
        _tin = _x.Dim(0, p);
        _tout = _y.Dim(0, p);
        if (_tout == 0)
        {
            return;
        }

        long work = _w.Dims[0] * _w.Dims[1] * _w.Dims[2];
        var (chunks, per) = OpHelpers.Split(_tout, work);
        _per = per;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int chunk)
    {
        long first = chunk * _per;
        long last = Math.Min(_tout, first + _per);
        long channels = _x.Dims[1];
        long outputs = _w.Dims[0];
        long perGroup = _w.Dims[1];
        long taps = _w.Dims[2];
        long outPerGroup = outputs / _groups;
        float* x = (float*)_x.P;
        float* w = (float*)_w.P;
        float* bias = _bias is null ? null : (float*)_bias.P;
        float* y = (float*)_y.P;
        for (long t = first; t < last; t++)
        {
            long origin = (t * _stride) - _padLeft;
            for (long o = 0; o < outputs; o++)
            {
                long group = o / outPerGroup;
                float acc = bias is null ? 0f : bias[o];
                float* wo = w + (o * perGroup * taps);
                for (long k = 0; k < taps; k++)
                {
                    long ti = origin + (k * _dilation);
                    if (_reflect)
                    {
                        ti = Reflect(ti, _tin);
                    }
                    else if (ti < 0 || ti >= _tin)
                    {
                        continue;
                    }

                    float* xr = x + (ti * channels) + (group * perGroup);
                    float* wk = wo + k;
                    for (long c = 0; c < perGroup; c++)
                    {
                        acc += xr[c] * wk[c * taps];
                    }
                }

                y[(t * outputs) + o] = acc;
            }
        }
    }

    private static long Reflect(long index, long length)
    {
        if (index < 0)
        {
            return -index;
        }

        return index >= length ? (2 * (length - 1)) - index : index;
    }
}
