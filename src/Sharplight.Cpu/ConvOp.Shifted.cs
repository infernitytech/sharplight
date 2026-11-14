namespace Sharplight.Cpu;

// The shifted form of stride-1 convolutions: no im2col. The input is copied once per run into a zero-padded buffer of
// Hp x Wp planes (Hp = HO + (KH-1)*DH, Wp = WO + (KW-1)*DW). Output pixel (oh, ow) becomes the virtual column v = oh*Wp + ow,
// and tap (fh, fw) of input channel c reads xpad[c][v + fh*DH*Wp + fw*DW]: for every tap the B rows of all input channels
// are the padded planes themselves, shifted by a constant and one plane apart, so the micro-kernel reads B in place. A KxK conv
// is KH*KW accumulated 1x1 GEMMs; the columns of the pad region (ow >= WO) are computed and dropped when the C block is
// written out. Tiles stay inside one image.
internal sealed unsafe partial class ConvOp
{
    private int _hp;
    private int _wp;
    private long _plane;
    private long _vcols;
    private long _tilesPerImage;
    private long* _tapOffsets;
    private long* _kOffsets;
    private float* _xpad;
    private long _xpadFloats;

    /// <summary>Geometry of the padded input and the per-tap read offsets.</summary>
    private void InitShifted()
    {
        int NR = Gemm.NR;
        _hp = _ho + (_kh - 1) * _dh;
        _wp = _wo + (_kw - 1) * _dw;
        _plane = (long)_hp * _wp;
        _vcols = (long)_ho * _wp;
        _tilesPerImage = (_vcols + NR - 1) / NR;
        int taps = _kh * _kw;
        _tapOffsets = (long*)Mem.Alloc(taps * 8L, false);
        for (int fh = 0; fh < _kh; fh++)
        {
            for (int fw = 0; fw < _kw; fw++)
            {
                _tapOffsets[fh * _kw + fw] = (long)fh * _dh * _wp + (long)fw * _dw;
            }
        }

        // B row of K index k = tap * CG + c: the tap's shift plus c planes.
        _kOffsets = (long*)Mem.Alloc((long)taps * _cg * 8, false);
        for (int t = 0; t < taps; t++)
        {
            for (int c = 0; c < _cg; c++)
            {
                _kOffsets[(long)t * _cg + c] = _tapOffsets[t] + c * _plane;
            }
        }
    }

    /// <summary>The dense [O][CG][KH*KW] weights reordered to [O][KH*KW][CG] (caller frees).</summary>
    private float* TapMajorWeights(float* wDense)
    {
        int taps = _kh * _kw;
        float* w = (float*)Mem.Alloc((long)_o * _kg * 4, false);
        for (long o = 0; o < _o; o++)
        {
            float* src = wDense + o * _kg;
            float* dst = w + o * _kg;
            for (int c = 0; c < _cg; c++)
            {
                for (int t = 0; t < taps; t++)
                {
                    dst[(long)t * _cg + c] = src[(long)c * taps + t];
                }
            }
        }

        return w;
    }

    /// <summary>
    /// Sizes the padded input buffer for this run. The last tile of a plane reads up to a tile plus the largest tap offset past
    /// the plane's start + vcols; past the last plane that is a zeroed tail (between planes it is the next plane's data, which only
    /// feeds dropped columns).
    /// </summary>
    private void PreparePadded()
    {
        long maxTap = _tapOffsets[_kh * _kw - 1];
        long tail = Math.Max(0, _tilesPerImage * Gemm.NR + maxTap - _plane) + 16;
        long floats = _n * _c * _plane + tail;
        if (floats > _xpadFloats)
        {
            Mem.Free((byte*)_xpad);
            _xpad = (float*)Mem.Alloc(floats * 4, true);
            _xpadFloats = floats;
        }
    }

    /// <summary>Copies plane <paramref name="task"/> (image * C + channel) of x into its zero-padded plane.</summary>
    private void PadBody(int task)
    {
        float* src = (float*)_x.P + (long)task * _h * _w;
        float* dst = _xpad + task * _plane;
        int top = Math.Min(_pt, _hp);
        int rows = Math.Clamp(_hp - _pt, 0, _h);
        int left = Math.Min(_pl, _wp);
        int cols = Math.Clamp(_wp - _pl, 0, _w);
        new Span<float>(dst, top * _wp).Clear();
        for (int r = 0; r < rows; r++)
        {
            float* d = dst + (long)(top + r) * _wp;
            new Span<float>(d, left).Clear();
            new ReadOnlySpan<float>(src + (long)r * _w, cols).CopyTo(new Span<float>(d + left, cols));
            new Span<float>(d + left + cols, _wp - left - cols).Clear();
        }

        new Span<float>(dst + (long)(top + rows) * _wp, (_hp - top - rows) * _wp).Clear();
    }

    /// <summary>
    /// Weight panels [p0, p1) over tiles [t0, t1) of group <paramref name="g"/> into the C block: K chunks span taps, each B row
    /// found through the per-k offset table.
    /// </summary>
    private void ShiftedGemm(int g, int p0, int p1, int t0, int t1, float* cbuf, long ldc)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        long kg = _kg;
        float* aGroup = _apack + (long)g * _mp * kg * MR;
        float* biasGroup = _biasPanels == null ? null : _biasPanels + (long)g * _mp * MR;
        for (int kb = 0; kb < _kg; kb += _kc)
        {
            int kc = Math.Min(_kc, _kg - kb);
            bool acc = kb > 0;
            for (int t = t0; t < t1; t++)
            {
                long img = t / _tilesPerImage;
                long v0 = (t - img * _tilesPerImage) * NR;
                float* b = _xpad + (img * _c + (long)g * _cg) * _plane + v0;
                float* ct = cbuf + (long)(t - t0) * NR;
                for (int p = p0; p < p1; p++)
                {
                    float* bias = acc || biasGroup == null ? null : biasGroup + (long)p * MR;
                    Gemm.TileGather(aGroup + ((long)p * kg + kb) * MR, b, _kOffsets + kb, ct + (long)(p - p0) * MR * ldc, ldc, kc, bias, acc);
                }
            }
        }
    }

    /// <summary>Writes the valid outputs of the C block (virtual columns with ow &lt; WO) to y through the epilogue.</summary>
    private void ShiftedCopyOut(int g, int p0, int p1, int t0, int t1, float* cbuf, long ldc)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        int rowStart = p0 * MR;
        int rowEnd = Math.Min(_mg, p1 * MR);
        float* y = (float*)_y.P;
        for (int t = t0; t < t1; t++)
        {
            long img = t / _tilesPerImage;
            long v0 = (t - img * _tilesPerImage) * NR;
            long vEnd = Math.Min(_vcols, v0 + NR);
            for (long v = v0; v < vEnd;)
            {
                long oh = v / _wp;
                long ow = v - oh * _wp;
                if (ow >= _wo)
                {
                    v = (oh + 1) * _wp;
                    continue;
                }

                int len = (int)Math.Min(_wo - ow, vEnd - v);
                float* src = cbuf + (long)(t - t0) * NR + (v - v0);
                float* dst = y + (img * _o + (long)g * _mg + rowStart) * _hw + oh * _wo + ow;
                for (int r = rowStart; r < rowEnd; r++)
                {
                    new ReadOnlySpan<float>(src, len).CopyTo(new Span<float>(dst, len));
                    _epilogue?.Apply(dst, dst - y, len);
                    src += ldc;
                    dst += _hw;
                }

                v += len;
            }
        }
    }
}
