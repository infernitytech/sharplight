using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 1-D transposed convolution of a dense f32 time-major signal <c>[T, C]</c> with a weight <c>[C, O/groups, K]</c> (see <c>StandardOps.ConvTranspose1d</c>),
/// computed as a gather per output position (parallel over output positions).
/// </summary>
internal sealed unsafe class ConvTranspose1dOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef? _bias;
    private readonly TRef _y;
    private readonly int _stride;
    private readonly int _cropLeft;
    private readonly int _groups;
    private readonly Action<int> _body;
    private long _tin;
    private long _tout;
    private long _per;

    /// <summary>Creates the op.</summary>
    public ConvTranspose1dOp(TRef x, TRef w, TRef? bias, TRef y, int stride, int cropLeft, int groups)
    {
        _x = x;
        _w = w;
        _bias = bias;
        _y = y;
        _stride = stride;
        _cropLeft = cropLeft;
        _groups = groups;
        _body = Body;
    }

    public override string Name => "ConvTranspose1d";

    public override void Run(RunParameters p)
    {
        _tin = _x.Dim(0, p);
        _tout = _y.Dim(0, p);
        if (_tout == 0)
        {
            return;
        }

        long work = _w.Dims[0] * _w.Dims[1] * _w.Dims[2] / _stride;
        var (chunks, per) = OpHelpers.Split(_tout, work);
        _per = per;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int chunk)
    {
        long first = chunk * _per;
        long last = Math.Min(_tout, first + _per);
        long channels = _x.Dims[0 + 1];
        long perGroup = channels / _groups;
        long outPerGroup = _w.Dims[1];
        long outputs = outPerGroup * _groups;
        long taps = _w.Dims[2];
        float* x = (float*)_x.P;
        float* w = (float*)_w.P;
        float* bias = _bias is null ? null : (float*)_bias.P;
        float* y = (float*)_y.P;
        for (long u = first; u < last; u++)
        {
            for (long o = 0; o < outputs; o++)
            {
                long group = o / outPerGroup;
                long oi = o % outPerGroup;
                float acc = bias is null ? 0f : bias[o];
                for (long k = 0; k < taps; k++)
                {
                    long num = u + _cropLeft - k;
                    if (num < 0 || num % _stride != 0)
                    {
                        continue;
                    }

                    long t = num / _stride;
                    if (t >= _tin)
                    {
                        continue;
                    }

                    float* xr = x + (t * channels) + (group * perGroup);
                    float* wk = w + (group * perGroup * outPerGroup * taps) + (oi * taps) + k;
                    for (long c = 0; c < perGroup; c++)
                    {
                        acc += xr[c] * wk[c * outPerGroup * taps];
                    }
                }

                y[(u * outputs) + o] = acc;
            }
        }
    }
}
