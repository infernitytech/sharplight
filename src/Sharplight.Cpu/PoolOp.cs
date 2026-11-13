using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>2-D max / average pooling over dense [N, C, H, W]; parallel over (n, c) planes.</summary>
internal sealed unsafe class PoolOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _max;
    private readonly bool _includePad;
    private readonly int _h;
    private readonly int _w;
    private readonly int _ho;
    private readonly int _wo;
    private readonly int _kh;
    private readonly int _kw;
    private readonly int _sh;
    private readonly int _sw;
    private readonly int _pt;
    private readonly int _pl;
    private readonly int _pb;
    private readonly int _pr;
    private readonly int _dh;
    private readonly int _dw;
    private readonly Action<int> _body;
    private long _planes;
    private long _per;
    public override string Name => _max ? "Pool.max" : "Pool.avg";

    public PoolOp(TRef x, TRef y, bool max, int[] kernel, int[] strides, int[] pads, int[] dil, bool countIncludePad)
    {
        _x = x;
        _y = y;
        _max = max;
        _includePad = countIncludePad;
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _kh = kernel[0];
        _kw = kernel[1];
        _sh = strides[0];
        _sw = strides[1];
        _pt = pads[0];
        _pl = pads[1];
        _pb = pads[2];
        _pr = pads[3];
        _dh = dil[0];
        _dw = dil[1];
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _planes = _x.Dim(0, p) * _x.Dims[1];
        if (_planes == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_planes, CpuParallel.Chunks(_planes * _ho * _wo * _kh * _kw, 16384));
        _per = (_planes + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_planes, s + _per);
        for (long pl = s; pl < e; pl++)
        {
            float* xp = (float*)_x.P + pl * _h * _w;
            float* yp = (float*)_y.P + pl * _ho * _wo;
            for (int oh = 0; oh < _ho; oh++)
            {
                int hs = oh * _sh - _pt;
                // valid kernel rows
                int fh0 = hs >= 0 ? 0 : (-hs + _dh - 1) / _dh;
                int fh1 = Math.Min(_kh, _h - 1 - hs < 0 ? 0 : (_h - 1 - hs) / _dh + 1);
                for (int ow = 0; ow < _wo; ow++)
                {
                    int ws = ow * _sw - _pl;
                    int fw0 = ws >= 0 ? 0 : (-ws + _dw - 1) / _dw;
                    int fw1 = Math.Min(_kw, _w - 1 - ws < 0 ? 0 : (_w - 1 - ws) / _dw + 1);
                    if (_max)
                    {
                        float m = float.NegativeInfinity;
                        for (int fh = fh0; fh < fh1; fh++)
                        {
                            float* r = xp + (long)(hs + fh * _dh) * _w + ws;
                            for (int fw = fw0; fw < fw1; fw++) { float v = r[fw * _dw]; if (v > m)
                                {
                                    m = v;
                                }
                            }
                        }
                        yp[oh * _wo + ow] = m;
                    }
                    else
                    {
                        float sum = 0;
                        for (int fh = fh0; fh < fh1; fh++)
                        {
                            float* r = xp + (long)(hs + fh * _dh) * _w + ws;
                            for (int fw = fw0; fw < fw1; fw++)
                            {
                                sum += r[fw * _dw];
                            }
                        }
                        long cnt;
                        if (_includePad)
                        {
                            // window clipped to the padded input (ceil_mode overhang is not counted)
                            long he = Math.Min(hs + _kh, _h + _pb);
                            long we = Math.Min(ws + _kw, _w + _pr);
                            cnt = (he - hs) * (we - ws);
                        }
                        else
                        {
                            cnt = (long)Math.Max(0, fh1 - fh0) * Math.Max(0, fw1 - fw0);
                        }

                        yp[oh * _wo + ow] = cnt > 0 ? sum / cnt : 0f;
                    }
                }
            }
        }
    }
}
