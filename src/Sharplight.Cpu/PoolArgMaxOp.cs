using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Flat input indices of 2-D max-pooling maxima over dense f32 [N, C, H, W] → i32 [N, C, HO, WO] (first maximum wins).</summary>
internal sealed unsafe class PoolArgMaxOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _columnMajor;
    private readonly int _h;
    private readonly int _w;
    private readonly int _ho;
    private readonly int _wo;
    private readonly int[] _kernel;
    private readonly int[] _strides;
    private readonly int[] _pads;
    private readonly int[] _dilations;

    public PoolArgMaxOp(TRef x, TRef y, int[] kernel, int[] strides, int[] pads, int[] dilations, bool columnMajor)
    {
        _x = x;
        _y = y;
        _kernel = kernel;
        _strides = strides;
        _pads = pads;
        _dilations = dilations;
        _columnMajor = columnMajor;
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
    }

    public override string Name => "PoolArgMax";

    public override void Run(RunParameters p)
    {
        long planes = _x.Dim(0, p) * _x.Dims[1];
        CpuParallel.For((int)Math.Min(planes, int.MaxValue), plane => Plane(plane));
    }

    private void Plane(long plane)
    {
        float* xp = (float*)_x.P + plane * _h * _w;
        int* yp = (int*)_y.P + plane * _ho * _wo;
        for (int oh = 0; oh < _ho; oh++)
        {
            int hs = oh * _strides[0] - _pads[0];
            for (int ow = 0; ow < _wo; ow++)
            {
                int ws = ow * _strides[1] - _pads[1];
                float best = float.NegativeInfinity;
                int bh = -1;
                int bw = -1;
                for (int kh = 0; kh < _kernel[0]; kh++)
                {
                    int ih = hs + kh * _dilations[0];
                    if ((uint)ih >= (uint)_h)
                    {
                        continue;
                    }

                    for (int kw = 0; kw < _kernel[1]; kw++)
                    {
                        int iw = ws + kw * _dilations[1];
                        if ((uint)iw < (uint)_w && (bh < 0 || xp[(long)ih * _w + iw] > best))
                        {
                            best = xp[(long)ih * _w + iw];
                            bh = ih;
                            bw = iw;
                        }
                    }
                }

                long local = bh < 0 ? 0 : _columnMajor ? bh + (long)bw * _h : (long)bh * _w + bw;
                yp[oh * _wo + ow] = (int)(plane * _h * _w + local);
            }
        }
    }
}
