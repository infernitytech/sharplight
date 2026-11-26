using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D max / average pooling on channels-last activations (<see cref="ChannelsLast"/>; the input may be a channels-last
/// view), vectorized over channels: each
/// output pixel takes the max or mean of its window's channel vectors. The global form (<see cref="Global"/>, a mean or
/// max over all of H and W) writes a dense [N, C] result, as a Reduce over the spatial axes does. Tasks are (image,
/// output row, channel block) triples, or (image, channel block) pairs for the global form.
/// </summary>
internal sealed unsafe class ChannelsLastPoolOp : CpuOp
{
    /// <summary>Tasks per thread aimed for.</summary>
    private const int TasksPerThread = 4;

    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool _max;
    private readonly bool _global;
    private readonly bool _includePad;
    private readonly int _c;
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
    private readonly long _ps;
    private readonly long _rs;
    private readonly long _is;
    private readonly Action<int> _body;

    // per-run state
    private long _rows;
    private int _cBlocks;
    private int _cPerBlock;

    /// <summary>Windowed pooling: <paramref name="x"/> and <paramref name="y"/> channels-last.</summary>
    public ChannelsLastPoolOp(TRef x, TRef y, bool max, int[] kernel, int[] strides, int[] pads, int[] dil, bool countIncludePad)
        : this(x, y, max, false)
    {
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
        _includePad = countIncludePad;
    }

    private ChannelsLastPoolOp(TRef x, TRef y, bool max, bool global)
    {
        _x = x;
        _y = y;
        _max = max;
        _global = global;
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ps = ChannelsLast.PixelStride(x);
        _rs = ChannelsLast.RowStride(x);
        _is = ChannelsLast.ImageStride(x);
        _ho = 1;
        _wo = 1;
        _body = Body;
    }

    /// <inheritdoc />
    public override string Name => (_global ? "Pool.cl.global" : "Pool.cl") + (_max ? ".max" : ".avg");

    /// <summary>Mean (or max) of channels-last <paramref name="x"/> over H and W into dense <paramref name="y"/> ([N, C, 1, 1] or [N, C]).</summary>
    public static ChannelsLastPoolOp Global(TRef x, TRef y, bool max) => new(x, y, max, true);

    /// <inheritdoc />
    public override void Run(RunParameters p)
    {
        long n = _x.Dim(0, p);
        _rows = n * (_global ? 1 : _ho);
        if (_rows == 0)
        {
            return;
        }

        int vectors = (_c + 15) / 16;
        long want = (long)TasksPerThread * CpuParallel.Threads;
        _cBlocks = (int)Math.Clamp((want + _rows - 1) / _rows, 1, vectors);
        _cPerBlock = (vectors + _cBlocks - 1) / _cBlocks * 16;
        _cBlocks = (_c + _cPerBlock - 1) / _cPerBlock;
        CpuParallel.For((int)(_rows * _cBlocks), _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        int cb = task % _cBlocks;
        long row = task / _cBlocks;
        int c0 = cb * _cPerBlock;
        int c1 = Math.Min(_c, c0 + _cPerBlock);
        int c = c0;
        if (CpuIsa.Use512)
        {
            for (; c + 16 <= c1; c += 16)
            {
                Channels<FloatLanes512>(row, c);
            }
        }

        if (CpuIsa.Use256)
        {
            for (; c + 8 <= c1; c += 8)
            {
                Channels<FloatLanes256>(row, c);
            }
        }
        else if (CpuIsa.Use128)
        {
            for (; c + 8 <= c1; c += 8)
            {
                Channels<FloatLanes128x2>(row, c);
            }
        }

        for (; c < c1; c++)
        {
            ScalarChannel(row, c);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Channels<TL>(long row, int c) where TL : struct, IFloatLanes<TL>
    {
        long cs = _c;
        float* x = (float*)_x.P;
        if (_global)
        {
            float* xp = x + row * _is + c;
            long pixels = (long)_h * _w;
            var acc = _max ? TL.Broadcast(float.NegativeInfinity) : TL.Zero;
            for (int ih = 0; ih < _h; ih++)
            {
                float* xr = xp + ih * _rs;
                for (int iw = 0; iw < _w; iw++)
                {
                    var v = TL.Load(xr + iw * _ps);
                    acc = _max ? TL.Max(acc, v) : TL.Add(acc, v);
                }
            }

            TL.Store(_max ? acc : TL.Multiply(acc, TL.Broadcast(1f / pixels)), (float*)_y.P + row * cs + c);
            return;
        }

        long n = row / _ho;
        int oh = (int)(row - n * _ho);
        int hs = oh * _sh - _pt;
        int fh0 = hs >= 0 ? 0 : (-hs + _dh - 1) / _dh;
        int fh1 = Math.Min(_kh, _h - 1 - hs < 0 ? 0 : (_h - 1 - hs) / _dh + 1);
        float* xn = x + n * _is + c;
        float* yr = (float*)_y.P + (n * _ho + oh) * _wo * cs + c;
        for (int ow = 0; ow < _wo; ow++)
        {
            int ws = ow * _sw - _pl;
            int fw0 = ws >= 0 ? 0 : (-ws + _dw - 1) / _dw;
            int fw1 = Math.Min(_kw, _w - 1 - ws < 0 ? 0 : (_w - 1 - ws) / _dw + 1);
            var acc = _max ? TL.Broadcast(float.NegativeInfinity) : TL.Zero;
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* r = xn + (hs + fh * _dh) * _rs + ws * _ps;
                for (int fw = fw0; fw < fw1; fw++)
                {
                    var v = TL.Load(r + fw * _dw * _ps);
                    acc = _max ? TL.Max(acc, v) : TL.Add(acc, v);
                }
            }

            if (!_max)
            {
                long cnt = Count(hs, ws, fh0, fh1, fw0, fw1);
                acc = cnt > 0 ? TL.Multiply(acc, TL.Broadcast(1f / cnt)) : TL.Zero;
            }

            TL.Store(acc, yr + ow * cs);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ScalarChannel(long row, int c)
    {
        long cs = _c;
        float* x = (float*)_x.P;
        if (_global)
        {
            float* xp = x + row * _is + c;
            long pixels = (long)_h * _w;
            float acc = _max ? float.NegativeInfinity : 0f;
            for (int ih = 0; ih < _h; ih++)
            {
                for (int iw = 0; iw < _w; iw++)
                {
                    float v = xp[ih * _rs + iw * _ps];
                    acc = _max ? Math.Max(acc, v) : acc + v;
                }
            }

            ((float*)_y.P)[row * cs + c] = _max ? acc : acc / pixels;
            return;
        }

        long n = row / _ho;
        int oh = (int)(row - n * _ho);
        int hs = oh * _sh - _pt;
        int fh0 = hs >= 0 ? 0 : (-hs + _dh - 1) / _dh;
        int fh1 = Math.Min(_kh, _h - 1 - hs < 0 ? 0 : (_h - 1 - hs) / _dh + 1);
        float* xn = x + n * _is + c;
        float* yr = (float*)_y.P + (n * _ho + oh) * _wo * cs + c;
        for (int ow = 0; ow < _wo; ow++)
        {
            int ws = ow * _sw - _pl;
            int fw0 = ws >= 0 ? 0 : (-ws + _dw - 1) / _dw;
            int fw1 = Math.Min(_kw, _w - 1 - ws < 0 ? 0 : (_w - 1 - ws) / _dw + 1);
            float acc = _max ? float.NegativeInfinity : 0f;
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* r = xn + (hs + fh * _dh) * _rs + ws * _ps;
                for (int fw = fw0; fw < fw1; fw++)
                {
                    float v = r[fw * _dw * _ps];
                    acc = _max ? Math.Max(acc, v) : acc + v;
                }
            }

            if (!_max)
            {
                long cnt = Count(hs, ws, fh0, fh1, fw0, fw1);
                acc = cnt > 0 ? acc / cnt : 0f;
            }

            yr[ow * cs] = acc;
        }
    }

    /// <summary>The divisor of an average window (as in <see cref="PoolOp"/>).</summary>
    private long Count(int hs, int ws, int fh0, int fh1, int fw0, int fw1)
    {
        if (_includePad)
        {
            long th = Math.Min(_kh, (_h + _pb - 1 - hs) / _dh + 1);
            long tw = Math.Min(_kw, (_w + _pr - 1 - ws) / _dw + 1);
            return th * tw;
        }

        return (long)Math.Max(0, fh1 - fh0) * Math.Max(0, fw1 - fw0);
    }
}
