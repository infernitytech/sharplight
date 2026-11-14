using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

// Depthwise convolution (groups == C): direct per-plane kernels with AVX-512 / AVX2 interior rows.
internal sealed unsafe partial class ConvOp
{
    /// <summary>Interior outputs per row from which the depthwise rows use the vector loops instead of one AXPY per tap.</summary>
    private const int DwVectorMinWidth = 16;

    private void DwBody(int chunk)
    {
        long s = chunk * _perChunk;
        long e = Math.Min(_planes, s + _perChunk);
        int mult = _o / _c;
        int taps = _kh * _kw;
        Vector512<float>* w512 = stackalloc Vector512<float>[CpuIsa.Use512 ? taps : 0];
        Vector256<float>* w256 = stackalloc Vector256<float>[CpuIsa.Use256 && !CpuIsa.Use512 ? taps : 0];
        // Interior outputs: every tap of the row is inside the input, so the vector loop needs no bounds checks.
        int owLo = Math.Min(_wo, (_pl + _sw - 1) / _sw);
        int lastTap = (_kw - 1) * _dw - _pl;
        int owHi = _w - 1 - lastTap < 0 ? owLo : Math.Clamp((_w - 1 - lastTap) / _sw + 1, owLo, _wo);
        for (long pl = s; pl < e; pl++)
        {
            long n = pl / _o;
            int o = (int)(pl % _o);
            int c = o / mult;
            float* xp = (float*)_x.P + (n * _c + c) * _h * _w;
            float* yp = (float*)_y.P + pl * _hw;
            float* wt = _wdw + (long)o * taps;
            float bv = _bias0 == null ? 0f : _bias0[o];
            for (int t = 0; t < taps; t++)
            {
                if (CpuIsa.Use512)
                {
                    w512[t] = Vector512.Create(wt[t]);
                }
                else if (CpuIsa.Use256)
                {
                    w256[t] = Vector256.Create(wt[t]);
                }
            }

            for (int oh = 0; oh < _ho; oh++)
            {
                float* yr = yp + (long)oh * _wo;
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

                float* row0 = xp + (long)(oh * _sh - _pt) * _w;
                if (owHi - owLo < DwVectorMinWidth || _sw > 2 || !CpuIsa.Use256)
                {
                    DwAxpyRow(yr, row0, wt, bv, fh0, fh1);
                }
                else
                {
                    int ow = owLo;
                    if (CpuIsa.Use512)
                    {
                        ow = DwRow512(yr, row0, w512, bv, fh0, fh1, owLo, owHi);
                    }

                    if (_sw == 1)
                    {
                        ow = CpuIsa.Use512 ? DwRow256(yr, row0, (Vector256<float>*)null, wt, bv, fh0, fh1, ow, owHi) : DwRow256(yr, row0, w256, wt, bv, fh0, fh1, ow, owHi);
                    }

                    DwScalar(yr, row0, wt, bv, fh0, fh1, 0, owLo);
                    DwScalar(yr, row0, wt, bv, fh0, fh1, ow, _wo);
                }

                _epilogue?.Apply(yr, yr - (float*)_y.P, _wo);
            }
        }
    }

    /// <summary>Interior outputs [owLo, owHi) in steps of 16 (stride 1, or 2 by an even-lane permute); returns the first one left.</summary>
    private int DwRow512(float* yr, float* row0, Vector512<float>* w, float bias, int fh0, int fh1, int owLo, int owHi)
    {
        var even = Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        int ow = owLo;
        // Stride 2 loads 32 inputs per tap; the last one is past the final output's tap and must still be inside the row.
        for (; ow + 16 <= owHi && (_sw == 1 || (ow + 15) * 2 - _pl + (_kw - 1) * _dw + 1 < _w); ow += 16)
        {
            var acc = Vector512.Create(bias);
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w + ow * _sw - _pl;
                for (int fw = 0; fw < _kw; fw++)
                {
                    float* p = xr + fw * _dw;
                    var v = _sw == 1 ? Vector512.Load(p) : Avx512F.PermuteVar16x32x2(Vector512.Load(p), even, Vector512.Load(p + 16));
                    acc = Vector512.FusedMultiplyAdd(v, w[fh * _kw + fw], acc);
                }
            }

            acc.Store(yr + ow);
        }

        return ow;
    }

    /// <summary>Interior outputs [owLo, owHi) in steps of 8, stride 1 (weights splatted in <paramref name="w"/>, or from <paramref name="wt"/> when null); returns the first one left.</summary>
    private int DwRow256(float* yr, float* row0, Vector256<float>* w, float* wt, float bias, int fh0, int fh1, int owLo, int owHi)
    {
        int ow = owLo;
        for (; ow + 8 <= owHi; ow += 8)
        {
            var acc = Vector256.Create(bias);
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w + ow - _pl;
                for (int fw = 0; fw < _kw; fw++)
                {
                    var wv = w != null ? w[fh * _kw + fw] : Vector256.Create(wt[fh * _kw + fw]);
                    acc = Vector256.FusedMultiplyAdd(Vector256.Load(xr + fw * _dw), wv, acc);
                }
            }

            acc.Store(yr + ow);
        }

        return ow;
    }

    /// <summary>
    /// One output row as one AXPY per tap over the valid output range (rows too narrow for the interior vector loop, strides
    /// above 2, or no 256-bit vectors).
    /// </summary>
    private void DwAxpyRow(float* yr, float* row0, float* wt, float bias, int fh0, int fh1)
    {
        new Span<float>(yr, _wo).Fill(bias);
        for (int fh = fh0; fh < fh1; fh++)
        {
            float* xr = row0 + (long)fh * _dh * _w;
            for (int fw = 0; fw < _kw; fw++)
            {
                float wv = wt[fh * _kw + fw];
                int off = fw * _dw - _pl;
                int lo = off >= 0 ? 0 : (-off + _sw - 1) / _sw;
                int hi = _w - 1 - off < 0 ? 0 : Math.Min(_wo, (_w - 1 - off) / _sw + 1);
                if (hi <= lo)
                {
                    continue;
                }

                if (_sw == 1)
                {
                    Simd.Axpy(yr + lo, xr + off + lo, wv, hi - lo);
                }
                else
                {
                    for (int ow = lo; ow < hi; ow++)
                    {
                        yr[ow] += wv * xr[ow * _sw + off];
                    }
                }
            }
        }
    }

    /// <summary>Outputs [ow0, ow1) of one row with bounds checks (borders and remainders of the vector loops).</summary>
    private void DwScalar(float* yr, float* row0, float* wt, float bias, int fh0, int fh1, int ow0, int ow1)
    {
        for (int ow = ow0; ow < ow1; ow++)
        {
            float acc = bias;
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w;
                for (int fw = 0; fw < _kw; fw++)
                {
                    int iw = ow * _sw - _pl + fw * _dw;
                    if ((uint)iw < (uint)_w)
                    {
                        acc = MathF.FusedMultiplyAdd(xr[iw], wt[fh * _kw + fw], acc);
                    }
                }
            }

            yr[ow] = acc;
        }
    }
}
