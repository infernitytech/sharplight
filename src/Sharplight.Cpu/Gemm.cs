using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// Register-blocked f32 GEMM micro-kernels shared by Conv (im2col) and BatchMatMul.
/// A is packed as [k][MR] panels, B as [k][NR] panels (or read in place with a row stride);
/// the kernel computes one MR x NR tile of C (row-major, leading dimension ldc):
/// C = (acc ? C : bias ? bias[row] : 0) + sum_k A[k][i] * B[k][j].
/// </summary>
internal static unsafe partial class Gemm
{
    public static readonly int MR;
    public static readonly int NR;
    public const int KC = 256;

    static Gemm()
    {
        if (CpuIsa.Use512)
        {
            MR = 8;
            NR = 32;
        }
        else if (CpuIsa.Use256 || CpuIsa.UseNeon)
        {
            MR = 6;
            NR = 16;
        }
        else
        {
            MR = 4;
            NR = 8;
        }
    }

    /// <summary>Full MR x NR tile.</summary>
    public static void Tile(float* a, float* b, long ldb, float* c, long ldc, int kc, float* bias, bool acc)
    {
        if (NR == 32)
        {
            K8x32(a, b, ldb, c, ldc, kc, bias, acc);
        }
        else if (NR == 16 && CpuIsa.UseNeon)
        {
            K6x16Neon(a, b, ldb, null, c, ldc, kc, bias, acc);
        }
        else if (NR == 16)
        {
            K6x16(a, b, ldb, c, ldc, kc, bias, acc);
        }
        else
        {
            KScalar(a, b, ldb, c, ldc, kc, bias, acc);
        }
    }

    /// <summary>
    /// Full MR x NR tile whose B row k is at <paramref name="b"/> + <paramref name="bOffsets"/>[k] (rows anywhere, NR contiguous
    /// floats each): implicit GEMMs that read B in place, e.g. the shifted taps of a convolution.
    /// </summary>
    public static void TileGather(float* a, float* b, long* bOffsets, float* c, long ldc, int kc, float* bias, bool acc)
    {
        if (NR == 32)
        {
            K8x32Gather(a, b, bOffsets, c, ldc, kc, bias, acc);
        }
        else if (NR == 16 && CpuIsa.UseNeon)
        {
            K6x16Neon(a, b, 0, bOffsets, c, ldc, kc, bias, acc);
        }
        else if (NR == 16)
        {
            K6x16Gather(a, b, bOffsets, c, ldc, kc, bias, acc);
        }
        else
        {
            // Scalar width: one k at a time through the strided kernel (ldb is unused for a single k row).
            for (int k = 0; k < kc; k++)
            {
                Tile(a + (long)k * MR, b + bOffsets[k], NR, c, ldc, 1, k == 0 ? bias : null, acc || k > 0);
            }
        }
    }

    /// <summary>Partial tile (rows &lt;= MR, cols &lt;= NR): computes into a scratch tile and copies the valid part.</summary>
    public static void TileEdge(float* a, float* b, long ldb, float* c, long ldc, int kc, float* bias, bool acc, int rows, int cols)
    {
        float* tmp = stackalloc float[MR * NR];
        if (acc)
        {
            for (int i = 0; i < MR; i++)
            {
                for (int j = 0; j < NR; j++)
                {
                    tmp[i * NR + j] = i < rows && j < cols ? c[i * ldc + j] : 0f;
                }
            }
        }

        Tile(a, b, ldb, tmp, NR, kc, bias, acc);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                c[i * ldc + j] = tmp[i * NR + j];
            }
        }
    }

    private static void KScalar(float* a, float* b, long ldb, float* c, long ldc, int kc, float* bias, bool acc)
    {
        const int mr = 4;
        const int nr = 8;
        float* t = stackalloc float[mr * nr];
        for (int i = 0; i < mr; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                t[i * nr + j] = acc ? c[i * ldc + j] : bias != null ? bias[i] : 0f;
            }
        }

        for (int k = 0; k < kc; k++)
        {
            for (int i = 0; i < mr; i++)
            {
                float av = a[k * mr + i];
                for (int j = 0; j < nr; j++)
                {
                    t[i * nr + j] += av * b[k * ldb + j];
                }
            }
        }
        for (int i = 0; i < mr; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                c[i * ldc + j] = t[i * nr + j];
            }
        }
    }

    /// <summary>dst[k*MR + i] = alpha * src[i*sm + k*sk] for i &lt; rows, zero for the remaining panel rows.</summary>
    public static void PackA(float* dst, float* src, long sm, long sk, int rows, int kc, float alpha)
    {
        int mr = MR;
        for (int k = 0; k < kc; k++)
        {
            float* d = dst + (long)k * mr;
            float* s = src + k * sk;
            int i = 0;
            for (; i < rows; i++)
            {
                d[i] = alpha * s[i * sm];
            }

            for (; i < mr; i++)
            {
                d[i] = 0f;
            }
        }
    }

    /// <summary>dst[k*NR + j] = src[k*sk + j*sn] for j &lt; cols, zero for the remaining tile columns.</summary>
    public static void PackB(float* dst, float* src, long sk, long sn, int kc, int cols)
    {
        int nr = NR;
        for (int k = 0; k < kc; k++)
        {
            float* d = dst + (long)k * nr;
            float* s = src + k * sk;
            if (sn == 1 && cols == nr)
            {
                if (nr == 32)
                {
                    Vector512.Load(s).Store(d);
                    Vector512.Load(s + 16).Store(d + 16);
                }
                else
                {
                    for (int j = 0; j < nr; j++)
                    {
                        d[j] = s[j];
                    }
                }
            }
            else
            {
                int j = 0;
                for (; j < cols; j++)
                {
                    d[j] = s[j * sn];
                }

                for (; j < nr; j++)
                {
                    d[j] = 0f;
                }
            }
        }
    }

    private static void K8x32(float* a, float* b, long ldb, float* c, long ldc, int kc, float* bias, bool acc)
    {
        Vector512<float> c00;
        Vector512<float> c01;
        Vector512<float> c10;
        Vector512<float> c11;
        Vector512<float> c20;
        Vector512<float> c21;
        Vector512<float> c30;
        Vector512<float> c31;
        Vector512<float> c40;
        Vector512<float> c41;
        Vector512<float> c50;
        Vector512<float> c51;
        Vector512<float> c60;
        Vector512<float> c61;
        Vector512<float> c70;
        Vector512<float> c71;
        if (acc)
        {
            c00 = Vector512.Load(c + 0 * ldc + 0);
            c01 = Vector512.Load(c + 0 * ldc + 16);
            c10 = Vector512.Load(c + 1 * ldc + 0);
            c11 = Vector512.Load(c + 1 * ldc + 16);
            c20 = Vector512.Load(c + 2 * ldc + 0);
            c21 = Vector512.Load(c + 2 * ldc + 16);
            c30 = Vector512.Load(c + 3 * ldc + 0);
            c31 = Vector512.Load(c + 3 * ldc + 16);
            c40 = Vector512.Load(c + 4 * ldc + 0);
            c41 = Vector512.Load(c + 4 * ldc + 16);
            c50 = Vector512.Load(c + 5 * ldc + 0);
            c51 = Vector512.Load(c + 5 * ldc + 16);
            c60 = Vector512.Load(c + 6 * ldc + 0);
            c61 = Vector512.Load(c + 6 * ldc + 16);
            c70 = Vector512.Load(c + 7 * ldc + 0);
            c71 = Vector512.Load(c + 7 * ldc + 16);
        }
        else if (bias != null)
        {
            c00 = Vector512.Create(bias[0]);
            c01 = c00;
            c10 = Vector512.Create(bias[1]);
            c11 = c10;
            c20 = Vector512.Create(bias[2]);
            c21 = c20;
            c30 = Vector512.Create(bias[3]);
            c31 = c30;
            c40 = Vector512.Create(bias[4]);
            c41 = c40;
            c50 = Vector512.Create(bias[5]);
            c51 = c50;
            c60 = Vector512.Create(bias[6]);
            c61 = c60;
            c70 = Vector512.Create(bias[7]);
            c71 = c70;
        }
        else
        {
            c00 = Vector512<float>.Zero;
            c01 = c00;
            c10 = c00;
            c11 = c00;
            c20 = c00;
            c21 = c00;
            c30 = c00;
            c31 = c00;
            c40 = c00;
            c41 = c00;
            c50 = c00;
            c51 = c00;
            c60 = c00;
            c61 = c00;
            c70 = c00;
            c71 = c00;
        }
        for (int k = 0; k < kc; k++)
        {
            var b0 = Vector512.Load(b + 0);
            var b1 = Vector512.Load(b + 16);
            var a0 = Vector512.Create(a[0]);
            c00 = Vector512.FusedMultiplyAdd(a0, b0, c00);
            c01 = Vector512.FusedMultiplyAdd(a0, b1, c01);
            var a1 = Vector512.Create(a[1]);
            c10 = Vector512.FusedMultiplyAdd(a1, b0, c10);
            c11 = Vector512.FusedMultiplyAdd(a1, b1, c11);
            var a2 = Vector512.Create(a[2]);
            c20 = Vector512.FusedMultiplyAdd(a2, b0, c20);
            c21 = Vector512.FusedMultiplyAdd(a2, b1, c21);
            var a3 = Vector512.Create(a[3]);
            c30 = Vector512.FusedMultiplyAdd(a3, b0, c30);
            c31 = Vector512.FusedMultiplyAdd(a3, b1, c31);
            var a4 = Vector512.Create(a[4]);
            c40 = Vector512.FusedMultiplyAdd(a4, b0, c40);
            c41 = Vector512.FusedMultiplyAdd(a4, b1, c41);
            var a5 = Vector512.Create(a[5]);
            c50 = Vector512.FusedMultiplyAdd(a5, b0, c50);
            c51 = Vector512.FusedMultiplyAdd(a5, b1, c51);
            var a6 = Vector512.Create(a[6]);
            c60 = Vector512.FusedMultiplyAdd(a6, b0, c60);
            c61 = Vector512.FusedMultiplyAdd(a6, b1, c61);
            var a7 = Vector512.Create(a[7]);
            c70 = Vector512.FusedMultiplyAdd(a7, b0, c70);
            c71 = Vector512.FusedMultiplyAdd(a7, b1, c71);
            a += 8;
            b += ldb;
        }
        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 16);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 16);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 16);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 16);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 16);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 16);
        c60.Store(c + 6 * ldc + 0);
        c61.Store(c + 6 * ldc + 16);
        c70.Store(c + 7 * ldc + 0);
        c71.Store(c + 7 * ldc + 16);
    }

    private static void K8x32Gather(float* a, float* b, long* bOffsets, float* c, long ldc, int kc, float* bias, bool acc)
    {
        Vector512<float> c00;
        Vector512<float> c01;
        Vector512<float> c10;
        Vector512<float> c11;
        Vector512<float> c20;
        Vector512<float> c21;
        Vector512<float> c30;
        Vector512<float> c31;
        Vector512<float> c40;
        Vector512<float> c41;
        Vector512<float> c50;
        Vector512<float> c51;
        Vector512<float> c60;
        Vector512<float> c61;
        Vector512<float> c70;
        Vector512<float> c71;
        if (acc)
        {
            c00 = Vector512.Load(c + 0 * ldc + 0);
            c01 = Vector512.Load(c + 0 * ldc + 16);
            c10 = Vector512.Load(c + 1 * ldc + 0);
            c11 = Vector512.Load(c + 1 * ldc + 16);
            c20 = Vector512.Load(c + 2 * ldc + 0);
            c21 = Vector512.Load(c + 2 * ldc + 16);
            c30 = Vector512.Load(c + 3 * ldc + 0);
            c31 = Vector512.Load(c + 3 * ldc + 16);
            c40 = Vector512.Load(c + 4 * ldc + 0);
            c41 = Vector512.Load(c + 4 * ldc + 16);
            c50 = Vector512.Load(c + 5 * ldc + 0);
            c51 = Vector512.Load(c + 5 * ldc + 16);
            c60 = Vector512.Load(c + 6 * ldc + 0);
            c61 = Vector512.Load(c + 6 * ldc + 16);
            c70 = Vector512.Load(c + 7 * ldc + 0);
            c71 = Vector512.Load(c + 7 * ldc + 16);
        }
        else if (bias != null)
        {
            c00 = Vector512.Create(bias[0]);
            c01 = c00;
            c10 = Vector512.Create(bias[1]);
            c11 = c10;
            c20 = Vector512.Create(bias[2]);
            c21 = c20;
            c30 = Vector512.Create(bias[3]);
            c31 = c30;
            c40 = Vector512.Create(bias[4]);
            c41 = c40;
            c50 = Vector512.Create(bias[5]);
            c51 = c50;
            c60 = Vector512.Create(bias[6]);
            c61 = c60;
            c70 = Vector512.Create(bias[7]);
            c71 = c70;
        }
        else
        {
            c00 = Vector512<float>.Zero;
            c01 = c00;
            c10 = c00;
            c11 = c00;
            c20 = c00;
            c21 = c00;
            c30 = c00;
            c31 = c00;
            c40 = c00;
            c41 = c00;
            c50 = c00;
            c51 = c00;
            c60 = c00;
            c61 = c00;
            c70 = c00;
            c71 = c00;
        }
        for (int k = 0; k < kc; k++)
        {
            float* bk = b + bOffsets[k];
            var b0 = Vector512.Load(bk);
            var b1 = Vector512.Load(bk + 16);
            var a0 = Vector512.Create(a[0]);
            c00 = Vector512.FusedMultiplyAdd(a0, b0, c00);
            c01 = Vector512.FusedMultiplyAdd(a0, b1, c01);
            var a1 = Vector512.Create(a[1]);
            c10 = Vector512.FusedMultiplyAdd(a1, b0, c10);
            c11 = Vector512.FusedMultiplyAdd(a1, b1, c11);
            var a2 = Vector512.Create(a[2]);
            c20 = Vector512.FusedMultiplyAdd(a2, b0, c20);
            c21 = Vector512.FusedMultiplyAdd(a2, b1, c21);
            var a3 = Vector512.Create(a[3]);
            c30 = Vector512.FusedMultiplyAdd(a3, b0, c30);
            c31 = Vector512.FusedMultiplyAdd(a3, b1, c31);
            var a4 = Vector512.Create(a[4]);
            c40 = Vector512.FusedMultiplyAdd(a4, b0, c40);
            c41 = Vector512.FusedMultiplyAdd(a4, b1, c41);
            var a5 = Vector512.Create(a[5]);
            c50 = Vector512.FusedMultiplyAdd(a5, b0, c50);
            c51 = Vector512.FusedMultiplyAdd(a5, b1, c51);
            var a6 = Vector512.Create(a[6]);
            c60 = Vector512.FusedMultiplyAdd(a6, b0, c60);
            c61 = Vector512.FusedMultiplyAdd(a6, b1, c61);
            var a7 = Vector512.Create(a[7]);
            c70 = Vector512.FusedMultiplyAdd(a7, b0, c70);
            c71 = Vector512.FusedMultiplyAdd(a7, b1, c71);
            a += 8;
        }
        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 16);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 16);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 16);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 16);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 16);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 16);
        c60.Store(c + 6 * ldc + 0);
        c61.Store(c + 6 * ldc + 16);
        c70.Store(c + 7 * ldc + 0);
        c71.Store(c + 7 * ldc + 16);
    }

    private static void K6x16(float* a, float* b, long ldb, float* c, long ldc, int kc, float* bias, bool acc)
    {
        Vector256<float> c00;
        Vector256<float> c01;
        Vector256<float> c10;
        Vector256<float> c11;
        Vector256<float> c20;
        Vector256<float> c21;
        Vector256<float> c30;
        Vector256<float> c31;
        Vector256<float> c40;
        Vector256<float> c41;
        Vector256<float> c50;
        Vector256<float> c51;
        if (acc)
        {
            c00 = Vector256.Load(c + 0 * ldc + 0);
            c01 = Vector256.Load(c + 0 * ldc + 8);
            c10 = Vector256.Load(c + 1 * ldc + 0);
            c11 = Vector256.Load(c + 1 * ldc + 8);
            c20 = Vector256.Load(c + 2 * ldc + 0);
            c21 = Vector256.Load(c + 2 * ldc + 8);
            c30 = Vector256.Load(c + 3 * ldc + 0);
            c31 = Vector256.Load(c + 3 * ldc + 8);
            c40 = Vector256.Load(c + 4 * ldc + 0);
            c41 = Vector256.Load(c + 4 * ldc + 8);
            c50 = Vector256.Load(c + 5 * ldc + 0);
            c51 = Vector256.Load(c + 5 * ldc + 8);
        }
        else if (bias != null)
        {
            c00 = Vector256.Create(bias[0]);
            c01 = c00;
            c10 = Vector256.Create(bias[1]);
            c11 = c10;
            c20 = Vector256.Create(bias[2]);
            c21 = c20;
            c30 = Vector256.Create(bias[3]);
            c31 = c30;
            c40 = Vector256.Create(bias[4]);
            c41 = c40;
            c50 = Vector256.Create(bias[5]);
            c51 = c50;
        }
        else
        {
            c00 = Vector256<float>.Zero;
            c01 = c00;
            c10 = c00;
            c11 = c00;
            c20 = c00;
            c21 = c00;
            c30 = c00;
            c31 = c00;
            c40 = c00;
            c41 = c00;
            c50 = c00;
            c51 = c00;
        }
        for (int k = 0; k < kc; k++)
        {
            var b0 = Vector256.Load(b + 0);
            var b1 = Vector256.Load(b + 8);
            var a0 = Vector256.Create(a[0]);
            c00 = Vector256.FusedMultiplyAdd(a0, b0, c00);
            c01 = Vector256.FusedMultiplyAdd(a0, b1, c01);
            var a1 = Vector256.Create(a[1]);
            c10 = Vector256.FusedMultiplyAdd(a1, b0, c10);
            c11 = Vector256.FusedMultiplyAdd(a1, b1, c11);
            var a2 = Vector256.Create(a[2]);
            c20 = Vector256.FusedMultiplyAdd(a2, b0, c20);
            c21 = Vector256.FusedMultiplyAdd(a2, b1, c21);
            var a3 = Vector256.Create(a[3]);
            c30 = Vector256.FusedMultiplyAdd(a3, b0, c30);
            c31 = Vector256.FusedMultiplyAdd(a3, b1, c31);
            var a4 = Vector256.Create(a[4]);
            c40 = Vector256.FusedMultiplyAdd(a4, b0, c40);
            c41 = Vector256.FusedMultiplyAdd(a4, b1, c41);
            var a5 = Vector256.Create(a[5]);
            c50 = Vector256.FusedMultiplyAdd(a5, b0, c50);
            c51 = Vector256.FusedMultiplyAdd(a5, b1, c51);
            a += 6;
            b += ldb;
        }
        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 8);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 8);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 8);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 8);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 8);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 8);
    }

    private static void K6x16Gather(float* a, float* b, long* bOffsets, float* c, long ldc, int kc, float* bias, bool acc)
    {
        Vector256<float> c00;
        Vector256<float> c01;
        Vector256<float> c10;
        Vector256<float> c11;
        Vector256<float> c20;
        Vector256<float> c21;
        Vector256<float> c30;
        Vector256<float> c31;
        Vector256<float> c40;
        Vector256<float> c41;
        Vector256<float> c50;
        Vector256<float> c51;
        if (acc)
        {
            c00 = Vector256.Load(c + 0 * ldc + 0);
            c01 = Vector256.Load(c + 0 * ldc + 8);
            c10 = Vector256.Load(c + 1 * ldc + 0);
            c11 = Vector256.Load(c + 1 * ldc + 8);
            c20 = Vector256.Load(c + 2 * ldc + 0);
            c21 = Vector256.Load(c + 2 * ldc + 8);
            c30 = Vector256.Load(c + 3 * ldc + 0);
            c31 = Vector256.Load(c + 3 * ldc + 8);
            c40 = Vector256.Load(c + 4 * ldc + 0);
            c41 = Vector256.Load(c + 4 * ldc + 8);
            c50 = Vector256.Load(c + 5 * ldc + 0);
            c51 = Vector256.Load(c + 5 * ldc + 8);
        }
        else if (bias != null)
        {
            c00 = Vector256.Create(bias[0]);
            c01 = c00;
            c10 = Vector256.Create(bias[1]);
            c11 = c10;
            c20 = Vector256.Create(bias[2]);
            c21 = c20;
            c30 = Vector256.Create(bias[3]);
            c31 = c30;
            c40 = Vector256.Create(bias[4]);
            c41 = c40;
            c50 = Vector256.Create(bias[5]);
            c51 = c50;
        }
        else
        {
            c00 = Vector256<float>.Zero;
            c01 = c00;
            c10 = c00;
            c11 = c00;
            c20 = c00;
            c21 = c00;
            c30 = c00;
            c31 = c00;
            c40 = c00;
            c41 = c00;
            c50 = c00;
            c51 = c00;
        }
        for (int k = 0; k < kc; k++)
        {
            float* bk = b + bOffsets[k];
            var b0 = Vector256.Load(bk);
            var b1 = Vector256.Load(bk + 8);
            var a0 = Vector256.Create(a[0]);
            c00 = Vector256.FusedMultiplyAdd(a0, b0, c00);
            c01 = Vector256.FusedMultiplyAdd(a0, b1, c01);
            var a1 = Vector256.Create(a[1]);
            c10 = Vector256.FusedMultiplyAdd(a1, b0, c10);
            c11 = Vector256.FusedMultiplyAdd(a1, b1, c11);
            var a2 = Vector256.Create(a[2]);
            c20 = Vector256.FusedMultiplyAdd(a2, b0, c20);
            c21 = Vector256.FusedMultiplyAdd(a2, b1, c21);
            var a3 = Vector256.Create(a[3]);
            c30 = Vector256.FusedMultiplyAdd(a3, b0, c30);
            c31 = Vector256.FusedMultiplyAdd(a3, b1, c31);
            var a4 = Vector256.Create(a[4]);
            c40 = Vector256.FusedMultiplyAdd(a4, b0, c40);
            c41 = Vector256.FusedMultiplyAdd(a4, b1, c41);
            var a5 = Vector256.Create(a[5]);
            c50 = Vector256.FusedMultiplyAdd(a5, b0, c50);
            c51 = Vector256.FusedMultiplyAdd(a5, b1, c51);
            a += 6;
        }
        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 8);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 8);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 8);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 8);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 8);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 8);
    }
}
