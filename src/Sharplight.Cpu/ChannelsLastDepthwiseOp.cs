using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Depthwise 2-D convolution (groups == C == O) on channels-last activations (<see cref="ChannelsLast"/>; the input may
/// be a channels-last view): the vector lanes
/// run over channels, so every tap is one vector load of the input and one of the packed weights ([tap][channel], zero
/// padded to the vector width). Interior pixels are computed four at a time, reusing each weight vector; the bias, a
/// residual addend in the output's layout and Relu / Clip are applied in registers. Tasks are (image, output row, channel
/// block) triples.
/// </summary>
internal sealed unsafe class ChannelsLastDepthwiseOp : CpuOp
{
    /// <summary>Tasks per thread aimed for.</summary>
    private const int TasksPerThread = 4;

    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly OutputEpilogue? _epilogue;
    private readonly int _c;
    private readonly int _cp;
    private readonly int _h;
    private readonly int _w;
    private readonly int _kh;
    private readonly int _kw;
    private readonly int _sh;
    private readonly int _sw;
    private readonly int _pt;
    private readonly int _pl;
    private readonly int _dh;
    private readonly int _dw;
    private readonly int _ho;
    private readonly int _wo;
    private readonly long _ps;
    private readonly long _rs;
    private readonly long _is;
    private readonly int _owLo;
    private readonly int _owHi;
    private readonly float* _wpack;     // [tap][_cp]
    private readonly float* _biasPad;   // [_cp]
    private readonly float _lo;
    private readonly float _hi;
    private readonly bool _inlineActivation;
    private readonly Action<int> _body;

    // per-run state
    private long _n;
    private int _cBlocks;
    private int _cPerBlock;

    public ChannelsLastDepthwiseOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int[] strides, int[] pads, int[] dil, OutputEpilogue? epilogue)
    {
        _x = x;
        _y = y;
        _bias = bias;
        _epilogue = epilogue;
        _c = (int)x.Dims[1];
        _cp = (_c + 15) & ~15;
        _ps = ChannelsLast.PixelStride(x);
        _rs = ChannelsLast.RowStride(x);
        _is = ChannelsLast.ImageStride(x);
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _kh = kh;
        _kw = kw;
        _sh = strides[0];
        _sw = strides[1];
        _pt = pads[0];
        _pl = pads[1];
        _dh = dil[0];
        _dw = dil[1];
        _body = Body;
        // Interior outputs: every horizontal tap inside the input.
        _owLo = Math.Min(_wo, (_pl + _sw - 1) / _sw);
        int lastTap = (_kw - 1) * _dw - _pl;
        _owHi = _w - 1 - lastTap < 0 ? _owLo : Math.Clamp((_w - 1 - lastTap) / _sw + 1, _owLo, _wo);
        int taps = kh * kw;
        _wpack = (float*)Mem.Alloc((long)taps * _cp * 4, true);
        for (int c = 0; c < _c; c++)
        {
            for (int t = 0; t < taps; t++)
            {
                _wpack[(long)t * _cp + c] = wDense[(long)c * taps + t];
            }
        }

        _biasPad = (float*)Mem.Alloc(_cp * 4L, true);
        _lo = float.NegativeInfinity;
        _hi = float.PositiveInfinity;
        _inlineActivation = epilogue?.Activation is null;
        if (epilogue?.Activation == UnaryKind.Relu)
        {
            _lo = 0;
            _inlineActivation = true;
        }
        else if (epilogue?.Activation == UnaryKind.Clip)
        {
            _lo = epilogue.Scale;
            _hi = epilogue.Bias;
            _inlineActivation = true;
        }
    }

    /// <inheritdoc />
    public override string Name => "Conv.cl.depthwise" + _epilogue?.Label;

    /// <inheritdoc />
    public override string Detail => $"{_n}x{_c}x{_h}x{_w} k{_kh}s{_sh} tasks {_n * _ho}x{_cBlocks}";

    /// <inheritdoc />
    public override void Dispose()
    {
        Mem.Free((byte*)_wpack);
        Mem.Free((byte*)_biasPad);
    }

    /// <inheritdoc />
    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        if (_n == 0)
        {
            return;
        }

        if (_bias is not null)
        {
            new ReadOnlySpan<float>((float*)_bias.P, _c).CopyTo(new Span<float>(_biasPad, _c));
        }

        long rows = _n * _ho;
        int vectors = _cp / 16;
        long want = (long)TasksPerThread * CpuParallel.Threads;
        _cBlocks = (int)Math.Clamp((want + rows - 1) / rows, 1, vectors);
        _cPerBlock = (vectors + _cBlocks - 1) / _cBlocks * 16;
        _cBlocks = (_cp + _cPerBlock - 1) / _cPerBlock;
        CpuParallel.For((int)(rows * _cBlocks), _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        int cb = task % _cBlocks;
        long row = task / _cBlocks;
        long n = row / _ho;
        int oh = (int)(row - n * _ho);
        int c0 = cb * _cPerBlock;
        int c1 = Math.Min(_c, c0 + _cPerBlock);
        int fh0 = 0;
        int fh1 = _kh;
        while (fh0 < _kh && (uint)(oh * _sh - _pt + fh0 * _dh) >= (uint)_h)
        {
            fh0++;
        }

        while (fh1 > fh0 && (uint)(oh * _sh - _pt + (fh1 - 1) * _dh) >= (uint)_h)
        {
            fh1--;
        }

        float* xr = (float*)_x.P + n * _is + (oh * _sh - _pt) * _rs;
        long yOff = (n * _ho + oh) * _wo * _c;
        float* yr = (float*)_y.P + yOff;
        float* rr = _epilogue?.Residual is { } r ? (float*)r.P + yOff : null;
        int c = c0;
        if (CpuIsa.Use512)
        {
            for (; c + 16 <= c1; c += 16)
            {
                Channels<FloatLanes512>(xr, yr, rr, c, fh0, fh1);
            }
        }

        if (CpuIsa.Use256)
        {
            for (; c + 8 <= c1; c += 8)
            {
                Channels<FloatLanes256>(xr, yr, rr, c, fh0, fh1);
            }
        }
        else if (CpuIsa.Use128)
        {
            for (; c + 8 <= c1; c += 8)
            {
                Channels<FloatLanes128x2>(xr, yr, rr, c, fh0, fh1);
            }
        }

        if (c < c1)
        {
            ScalarChannels(xr, yr, rr, c, c1, fh0, fh1);
        }

        if (!_inlineActivation)
        {
            for (int ow = 0; ow < _wo; ow++)
            {
                float* d = yr + (long)ow * _c + c0;
                ScalarMath.Unary(_epilogue!.Activation!.Value, d, d, c1 - c0, _epilogue.Scale, _epilogue.Bias);
            }
        }
    }

    /// <summary>One vector of channels from <paramref name="c"/> over the whole output row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Channels<TL>(float* xr, float* yr, float* rr, int c, int fh0, int fh1) where TL : struct, IFloatLanes<TL>
    {
        var bias = TL.Load(_biasPad + c);
        var lo = TL.Broadcast(_lo);
        var hi = TL.Broadcast(_hi);
        long cs = _c;
        float* w = _wpack + c;
        int ow = 0;
        for (; ow < _owLo; ow++)
        {
            Edge<TL>(xr, yr, rr, c, fh0, fh1, ow, bias, lo, hi);
        }

        for (; ow + 4 <= _owHi; ow += 4)
        {
            var a0 = bias;
            var a1 = bias;
            var a2 = bias;
            var a3 = bias;
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xp = xr + fh * _dh * _rs + ((long)ow * _sw - _pl) * _ps + c;
                float* wt = w + (long)fh * _kw * _cp;
                long step = _sw * _ps;
                for (int fw = 0; fw < _kw; fw++)
                {
                    var wv = TL.Load(wt + (long)fw * _cp);
                    float* xq = xp + fw * _dw * _ps;
                    a0 = TL.Fma(TL.Load(xq), wv, a0);
                    a1 = TL.Fma(TL.Load(xq + step), wv, a1);
                    a2 = TL.Fma(TL.Load(xq + 2 * step), wv, a2);
                    a3 = TL.Fma(TL.Load(xq + 3 * step), wv, a3);
                }
            }

            float* d = yr + (long)ow * cs + c;
            Store(a0, d, rr == null ? null : rr + (long)ow * cs + c, lo, hi);
            Store(a1, d + cs, rr == null ? null : rr + (long)(ow + 1) * cs + c, lo, hi);
            Store(a2, d + 2 * cs, rr == null ? null : rr + (long)(ow + 2) * cs + c, lo, hi);
            Store(a3, d + 3 * cs, rr == null ? null : rr + (long)(ow + 3) * cs + c, lo, hi);
        }

        for (; ow < _wo; ow++)
        {
            Edge<TL>(xr, yr, rr, c, fh0, fh1, ow, bias, lo, hi);
        }
    }

    /// <summary>One output pixel with bounds-checked horizontal taps.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Edge<TL>(float* xr, float* yr, float* rr, int c, int fh0, int fh1, int ow, TL bias, TL lo, TL hi) where TL : struct, IFloatLanes<TL>
    {
        long cs = _c;
        var a = bias;
        for (int fh = fh0; fh < fh1; fh++)
        {
            float* xrow = xr + fh * _dh * _rs + c;
            float* wt = _wpack + (long)fh * _kw * _cp + c;
            for (int fw = 0; fw < _kw; fw++)
            {
                int iw = ow * _sw - _pl + fw * _dw;
                if ((uint)iw < (uint)_w)
                {
                    a = TL.Fma(TL.Load(xrow + iw * _ps), TL.Load(wt + (long)fw * _cp), a);
                }
            }
        }

        Store(a, yr + ow * cs + c, rr == null ? null : rr + ow * cs + c, lo, hi);
    }

    private static void Store<TL>(TL v, float* d, float* residual, TL lo, TL hi) where TL : struct, IFloatLanes<TL>
    {
        if (residual != null)
        {
            v = TL.Add(v, TL.Load(residual));
        }

        TL.Store(TL.Min(TL.Max(v, lo), hi), d);
    }

    /// <summary>Channels [c0, c1) (fewer than a vector) one at a time.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ScalarChannels(float* xr, float* yr, float* rr, int c0, int c1, int fh0, int fh1)
    {
        long cs = _c;
        for (int ow = 0; ow < _wo; ow++)
        {
            for (int c = c0; c < c1; c++)
            {
                float a = _biasPad[c];
                for (int fh = fh0; fh < fh1; fh++)
                {
                    float* xrow = xr + fh * _dh * _rs + c;
                    float* wt = _wpack + (long)fh * _kw * _cp + c;
                    for (int fw = 0; fw < _kw; fw++)
                    {
                        int iw = ow * _sw - _pl + fw * _dw;
                        if ((uint)iw < (uint)_w)
                        {
                            a += xrow[iw * _ps] * wt[(long)fw * _cp];
                        }
                    }
                }

                if (rr != null)
                {
                    a += rr[ow * cs + c];
                }

                yr[ow * cs + c] = Math.Min(Math.Max(a, _lo), _hi);
            }
        }
    }
}
