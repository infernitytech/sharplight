using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Deformable 2-D convolution: per batch item the sampled columns [C·kh·kw, HO·WO] (bilinear, zero outside, times the mask) are built
/// first, then every group's output rows are the weight rows times its column block, plus bias.
/// </summary>
internal sealed unsafe class DeformConvOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _offset;
    private readonly TRef _bias;
    private readonly TRef? _mask;
    private readonly TRef _y;
    private readonly int[] _strides;
    private readonly int[] _pads;
    private readonly int[] _dilations;
    private readonly int _groups;
    private readonly int _offsetGroups;
    private readonly int _c;
    private readonly int _h;
    private readonly int _wd;
    private readonly int _o;
    private readonly int _kh;
    private readonly int _kw;
    private readonly int _ho;
    private readonly int _wo;
    private float[] _columns = [];

    public DeformConvOp(TRef x, TRef w, TRef offset, TRef bias, TRef? mask, TRef y, int[] strides, int[] pads, int[] dilations, int groups, int offsetGroups)
    {
        _x = x;
        _w = w;
        _offset = offset;
        _bias = bias;
        _mask = mask;
        _y = y;
        _strides = strides;
        _pads = pads;
        _dilations = dilations;
        _groups = groups;
        _offsetGroups = offsetGroups;
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _wd = (int)x.Dims[3];
        _o = (int)w.Dims[0];
        _kh = (int)w.Dims[2];
        _kw = (int)w.Dims[3];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
    }

    public override string Name => "DeformConv";

    public override void Run(RunParameters p)
    {
        long batches = _x.Dim(0, p);
        int taps = _kh * _kw;
        int points = _ho * _wo;
        long columnCount = (long)_c * taps * points;
        if (_columns.Length < columnCount)
        {
            _columns = new float[columnCount];
        }

        fixed (float* columns = _columns)
        {
            float* cols = columns;
            for (long n = 0; n < batches; n++)
            {
                long batch = n;
                CpuParallel.For(_c * taps, row => Sample(batch, row / taps, row % taps, cols + (long)row * points));
                CpuParallel.For(_o, o => Contract(batch, o, cols));
            }
        }
    }

    /// <summary>Column row (channel, tap): the deformed bilinear samples of one input channel for every output position.</summary>
    private void Sample(long n, int channel, int tap, float* row)
    {
        int group = channel / (_c / _offsetGroups);
        int i = tap / _kw;
        int j = tap % _kw;
        int points = _ho * _wo;
        float* plane = (float*)_x.P + (n * _c + channel) * _h * _wd;
        float* dy = (float*)_offset.P + ((n * _offsetGroups + group) * _kh * _kw + tap) * 2L * points;
        float* dx = dy + points;
        float* mask = _mask is null ? null : (float*)_mask.P + ((n * _offsetGroups + group) * _kh * _kw + tap) * (long)points;
        for (int oh = 0; oh < _ho; oh++)
        {
            for (int ow = 0; ow < _wo; ow++)
            {
                int q = oh * _wo + ow;
                float fy = oh * _strides[0] - _pads[0] + i * _dilations[0] + dy[q];
                float fx = ow * _strides[1] - _pads[1] + j * _dilations[1] + dx[q];
                float y0 = MathF.Floor(fy);
                float x0 = MathF.Floor(fx);
                int yi = (int)y0;
                int xi = (int)x0;
                float ly = fy - y0;
                float lx = fx - x0;
                float v = (1 - ly) * ((1 - lx) * Px(plane, yi, xi) + lx * Px(plane, yi, xi + 1))
                    + ly * ((1 - lx) * Px(plane, yi + 1, xi) + lx * Px(plane, yi + 1, xi + 1));
                row[q] = mask is null ? v : v * mask[q];
            }
        }
    }

    private float Px(float* plane, int r, int c) => (uint)r < (uint)_h && (uint)c < (uint)_wd ? plane[(long)r * _wd + c] : 0f;

    /// <summary>Output channel o: bias plus its weight row times the column block of its group.</summary>
    private void Contract(long n, int o, float* columns)
    {
        int points = _ho * _wo;
        int taps = _kh * _kw;
        int channelsPerGroup = _c / _groups;
        int group = o / (_o / _groups);
        int depth = channelsPerGroup * taps;
        float* weights = (float*)_w.P + (long)o * depth;
        float* block = columns + (long)group * depth * points;
        float* y = (float*)_y.P + (n * _o + o) * points;
        float bias = ((float*)_bias.P)[o];
        for (int q = 0; q < points; q++)
        {
            y[q] = bias;
        }

        for (int k = 0; k < depth; k++)
        {
            float weight = weights[k];
            float* row = block + (long)k * points;
            for (int q = 0; q < points; q++)
            {
                y[q] += weight * row[q];
            }
        }
    }
}
