using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D convolution of a dense f32 time-major feature map <c>[T, C, W]</c> with a weight <c>[O, C/groups, KT, KW]</c> (see
/// <c>StandardOps.SequenceConv2d</c>); parallel over output time steps, zero padding handled by clipping the taps.
/// </summary>
internal sealed unsafe class SequenceConv2dOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef? _bias;
    private readonly TRef _y;
    private readonly int _strideT;
    private readonly int _strideW;
    private readonly int _padT;
    private readonly int _padW;
    private readonly int _groups;
    private readonly Action<int> _body;
    private long _tin;
    private long _tout;
    private long _per;

    /// <summary>Creates the op.</summary>
    public SequenceConv2dOp(TRef x, TRef w, TRef? bias, TRef y, int strideT, int strideW, int padT, int padW, int groups)
    {
        _x = x;
        _w = w;
        _bias = bias;
        _y = y;
        _strideT = strideT;
        _strideW = strideW;
        _padT = padT;
        _padW = padW;
        _groups = groups;
        _body = Body;
    }

    public override string Name => "SequenceConv2d";

    public override void Run(RunParameters p)
    {
        _tin = _x.Dim(0, p);
        _tout = _y.Dim(0, p);
        if (_tout == 0)
        {
            return;
        }

        long work = _w.Dims[0] * _w.Dims[1] * _w.Dims[2] * _w.Dims[3] * _y.Dims[2];
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
        long width = _x.Dims[2];
        long outputs = _w.Dims[0];
        long perGroup = _w.Dims[1];
        long kt = _w.Dims[2];
        long kw = _w.Dims[3];
        long outWidth = _y.Dims[2];
        long outPerGroup = outputs / _groups;
        float* x = (float*)_x.P;
        float* w = (float*)_w.P;
        float* bias = _bias is null ? null : (float*)_bias.P;
        float* y = (float*)_y.P;
        for (long t = first; t < last; t++)
        {
            long originT = (t * _strideT) - _padT;
            for (long o = 0; o < outputs; o++)
            {
                long group = o / outPerGroup;
                float* wo = w + (o * perGroup * kt * kw);
                float* yo = y + (((t * outputs) + o) * outWidth);
                float initial = bias is null ? 0f : bias[o];
                for (long u = 0; u < outWidth; u++)
                {
                    yo[u] = initial;
                }

                for (long c = 0; c < perGroup; c++)
                {
                    long channel = (group * perGroup) + c;
                    for (long i = 0; i < kt; i++)
                    {
                        long ti = originT + i;
                        if (ti < 0 || ti >= _tin)
                        {
                            continue;
                        }

                        float* xr = x + (((ti * channels) + channel) * width);
                        float* wr = wo + (((c * kt) + i) * kw);
                        for (long j = 0; j < kw; j++)
                        {
                            float tap = wr[j];
                            for (long u = 0; u < outWidth; u++)
                            {
                                long wi = (u * _strideW) - _padW + j;
                                if (wi >= 0 && wi < width)
                                {
                                    yo[u] += tap * xr[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
