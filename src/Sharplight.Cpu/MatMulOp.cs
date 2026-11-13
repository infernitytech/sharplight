using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>y = x · Wᵀ with on-the-fly dequantization of W. x rows are f32; W is [N, K].</summary>
internal sealed unsafe class MatMulOp : CpuOp
{
    private const int KB = 1024;
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _y;
    private readonly WeightFormat _fmt;
    private readonly long _n;
    private readonly long _k;
    private readonly long _rowBytes;
    private readonly bool _fused;
    private readonly float** _xrows;
    private readonly Action<int> _body;
    [ThreadStatic] private static float[]? s_acc;
    private long _m;
    private long _groups;
    private long _perChunk;
    public override string Name => $"MatMul.{_fmt}";

    public MatMulOp(TRef x, TRef w, TRef y, WeightFormat fmt)
    {
        _x = x;
        _w = w;
        _y = y;
        _fmt = fmt;
        _n = w.Dims[0];
        _k = w.Dims[1];
        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        if (_k % Quant.BlockSize(fmt) != 0)
        {
            throw new ArgumentException($"K={_k} is not a multiple of the {fmt} block size.");
        }

        long maxRows = Math.Max(1, x.MaxElements / Math.Max(1, _k));
        _fused = Quant.HasFusedKernel(fmt);
        _xrows = (float**)Mem.Alloc(maxRows * sizeof(float*), false);
        _body = Body;
    }

    public override void Dispose() => Mem.Free((byte*)_xrows);

    public override void Run(RunParameters p)
    {
        _m = _x.Rows(p);
        if (_m == 0)
        {
            return;
        }

        for (long r = 0; r < _m; r++)
        {
            _xrows[r] = _x.F(r);
        }

        _groups = (_n + 3) / 4;
        int chunks = (int)Math.Min(_groups, CpuParallel.Threads * 8L);
        _perChunk = (_groups + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long g0 = c * _perChunk;
        long g1 = Math.Min(_groups, g0 + _perChunk);
        long m = _m;
        var accArr = s_acc;
        if (accArr is null || accArr.Length < m * 4)
        {
            s_acc = accArr = new float[Math.Max(m * 4, 4096)];
        }

        float* wbuf = stackalloc float[4 * KB];
        float* res = stackalloc float[8];
        float** wp = stackalloc float*[4];
        byte** rows = stackalloc byte*[4];
        int bs = Quant.BlockSize(_fmt);
        int bb = Quant.BlockBytes(_fmt);
        fixed (float* acc = accArr)
        {
            for (long g = g0; g < g1; g++)
            {
                long n0 = g * 4;
                int nr = (int)Math.Min(4, _n - n0);
                for (int r = 0; r < 4; r++)
                {
                    rows[r] = _w.P + Math.Min(n0 + r, _n - 1) * _rowBytes;
                }

                if (m == 1 && _fused)
                {
                    Quant.DotRows4(_fmt, _xrows[0], rows, _k / bs, res);
                    float* y1 = (float*)_y.P + n0;
                    for (int r = 0; r < nr; r++)
                    {
                        y1[r] = res[r];
                    }

                    continue;
                }
                new Span<float>(acc, (int)(m * 4)).Clear();
                for (long kb = 0; kb < _k; kb += KB)
                {
                    long len = Math.Min(KB, _k - kb);
                    for (int r = 0; r < 4; r++)
                    {
                        if (_fmt == WeightFormat.F32)
                        {
                            wp[r] = (float*)rows[r] + kb;
                        }
                        else
                        {
                            Quant.Dequantize(_fmt, rows[r] + kb / bs * bb, len, wbuf + r * KB);
                            wp[r] = wbuf + r * KB;
                        }
                    }
                    long t = 0;
                    for (; t + 2 <= m; t += 2)
                    {
                        Simd.Dot4x2(_xrows[t] + kb, _xrows[t + 1] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        float* a1 = a0 + 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                        a1[0] += res[4];
                        a1[1] += res[5];
                        a1[2] += res[6];
                        a1[3] += res[7];
                    }
                    if (t < m)
                    {
                        Simd.Dot4(_xrows[t] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                    }
                }
                for (long t = 0; t < m; t++)
                {
                    float* y = (float*)_y.P + t * _n + n0;
                    for (int r = 0; r < nr; r++)
                    {
                        y[r] = acc[t * 4 + r];
                    }
                }
            }
        }
    }
}
