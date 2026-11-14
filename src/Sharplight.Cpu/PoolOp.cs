using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D max / average pooling over dense [N, C, H, W]. Undilated max pooling runs per block of output rows: a vectorized
/// max over the window's input rows into a padded row buffer, then the horizontal window (vectorized for stride 1), in
/// parallel over (plane, row block). The general form is parallel over (n, c) planes.
/// </summary>
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
    private readonly Action<int> _rowBody;
    private readonly bool _rowForm;
    private readonly int _rowWidth;
    private long _planes;
    private long _per;
    private int _rowBlocks;
    private int _rowsPerBlock;
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
        _rowBody = RowBody;
        _rowForm = max && _dh == 1 && _dw == 1;
        _rowWidth = Math.Max((_wo - 1) * _sw + _kw, _pl + _w) + Vector256<float>.Count;
    }

    public override void Run(RunParameters p)
    {
        _planes = _x.Dim(0, p) * _x.Dims[1];
        if (_planes == 0)
        {
            return;
        }

        if (_rowForm)
        {
            long work = _planes * _ho * _wo * _kh * _kw;
            long tasks = Math.Max(1, Math.Min(CpuParallel.Threads * 4L, work / 32768));
            _rowBlocks = (int)Math.Clamp((tasks + _planes - 1) / _planes, 1, _ho);
            _rowsPerBlock = (_ho + _rowBlocks - 1) / _rowBlocks;
            _rowBlocks = (_ho + _rowsPerBlock - 1) / _rowsPerBlock;
            CpuParallel.For((int)(_planes * _rowBlocks), _rowBody);
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
                            // taps inside the padded input (ceil_mode overhang is not counted)
                            long th = Math.Min(_kh, (_h + _pb - 1 - hs) / _dh + 1);
                            long tw = Math.Min(_kw, (_w + _pr - 1 - ws) / _dw + 1);
                            cnt = th * tw;
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

    /// <summary>Undilated max pooling of one block of output rows of one plane.</summary>
    private void RowBody(int task)
    {
        long plane = task / _rowBlocks;
        int oh0 = task % _rowBlocks * _rowsPerBlock;
        int oh1 = Math.Min(_ho, oh0 + _rowsPerBlock);
        float* xp = (float*)_x.P + plane * _h * _w;
        float* yp = (float*)_y.P + plane * _ho * _wo;
        float* vm = (float*)ThreadScratch.Get(ThreadScratch.Worker, (long)_rowWidth * 4);
        new Span<float>(vm, _rowWidth).Fill(float.NegativeInfinity);
        float* vx = vm + _pl;
        int vec = Vector256<float>.Count;
        for (int oh = oh0; oh < oh1; oh++)
        {
            int hs = oh * _sh - _pt;
            int r0 = Math.Max(0, hs);
            int r1 = Math.Min(_h, hs + _kh);
            float* yr = yp + (long)oh * _wo;
            if (r0 >= r1)
            {
                new Span<float>(yr, _wo).Fill(float.NegativeInfinity);
                continue;
            }

            // Vertical max of the window's input rows into vx[0, W).
            float* first = xp + (long)r0 * _w;
            int i = 0;
            for (; i + vec <= _w; i += vec)
            {
                var m = Vector256.Load(first + i);
                for (int r = r0 + 1; r < r1; r++)
                {
                    m = Vector256.MaxNative(Vector256.Load(xp + (long)r * _w + i), m);
                }

                m.Store(vx + i);
            }

            for (; i < _w; i++)
            {
                float m = first[i];
                for (int r = r0 + 1; r < r1; r++)
                {
                    m = float.MaxNative(xp[(long)r * _w + i], m);
                }

                vx[i] = m;
            }

            // Horizontal window over the padded row (-inf outside the input).
            int ow = 0;
            if (_sw == 1)
            {
                for (; ow + vec <= _wo; ow += vec)
                {
                    var m = Vector256.Load(vm + ow);
                    for (int f = 1; f < _kw; f++)
                    {
                        m = Vector256.MaxNative(Vector256.Load(vm + ow + f), m);
                    }

                    m.Store(yr + ow);
                }
            }

            for (; ow < _wo; ow++)
            {
                float* w = vm + (long)ow * _sw;
                float m = w[0];
                for (int f = 1; f < _kw; f++)
                {
                    m = float.MaxNative(w[f], m);
                }

                yr[ow] = m;
            }
        }
    }
}
