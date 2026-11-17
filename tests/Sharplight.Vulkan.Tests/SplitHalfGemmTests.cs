using System.Runtime.InteropServices;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The split-f16 tensor-core GEMM (<see cref="CompileOptions.SplitHalfMatMul"/>, the strict-precision default for f32 weights):
/// its error against an f64 reference is at most that of the f32 SIMT GEMM, for values far outside f16's range too, and far
/// below single f16 products (reduced precision).
/// </summary>
public sealed class SplitHalfGemmTests
{
    /// <summary>
    /// Encoder-like shapes (weights ~0.03, activations ~1), shapes off the tile grid (split-K, partial tiles), and value
    /// ranges f16 cannot hold: activations up to 1e6, weights around 1e-7, rows spanning 2^±20.
    /// </summary>
    [Theory]
    [InlineData(1024, 768, 2048, 0.03f, 1f, false)]
    [InlineData(1000, 384, 1536, 0.03f, 1f, false)]
    [InlineData(1576, 3072, 768, 0.03f, 1f, false)]
    [InlineData(197, 768, 2304, 0.03f, 1f, false)]
    [InlineData(300, 96, 200, 1f, 1f, false)]
    [InlineData(1024, 768, 1024, 1e-7f, 1e6f, false)]
    [InlineData(1024, 768, 1024, 1e3f, 1e-6f, false)]
    [InlineData(1024, 768, 1024, 0.03f, 1f, true)]
    public void ErrorAtMostF32(int rows, int k, int n, float weightScale, float activationScale, bool wideRange)
    {
        Gpu.Require();
        var x = Rnd.F((long)rows * k, 1, activationScale);
        var w = Rnd.F((long)n * k, 2, weightScale);
        if (wideRange)
        {
            // Every value scaled by 2^e with e in [-20, 20], and a zero row.
            var rng = new Random(3);
            for (long i = 0; i < x.Length; i++)
            {
                x[i] = MathF.ScaleB(x[i], rng.Next(-20, 21));
            }

            for (long i = 0; i < w.Length; i++)
            {
                w[i] = MathF.ScaleB(w[i], rng.Next(-20, 21));
            }

            Array.Clear(x, 0, k);
        }

        var expected = Reference(x, w, rows, k, n);
        double strict = RelativeRms(Run(x, w, rows, k, n, split: false, reduced: false), expected);
        var splitResult = Run(x, w, rows, k, n, split: true, reduced: false);
        double split = RelativeRms(splitResult, expected);
        double reduced = RelativeRms(Run(x, w, rows, k, n, split: false, reduced: true), expected);
        TestContext.Current.TestOutputHelper?.WriteLine($"[{rows} x {k}] x [{k} -> {n}] w~{weightScale} x~{activationScale}{(wideRange ? " wide" : "")}: relative rms f32 SIMT {strict:E2}, split-f16 {split:E2}, f16 {reduced:E2}");
        Assert.True(strict < 1e-6, $"strict {strict:E2}");
        Assert.True(split <= strict * 1.1, $"split-f16 {split:E2} vs f32 SIMT {strict:E2}");
        Assert.All(splitResult, v => Assert.True(float.IsFinite(v)));
        if (wideRange)
        {
            Assert.All(splitResult.AsSpan(0, n).ToArray(), v => Assert.Equal(0f, v));
        }
    }

    /// <summary>
    /// Fast matrix precision (<see cref="MatrixPrecision.Fast"/>): one f16 product of row-scaled inputs per multiply-add. Its
    /// error is that of rounding both inputs to 11 significant bits (TF32's class, about 2e-4 relative RMS), on the ranges
    /// plain f16 products cannot hold too.
    /// </summary>
    [Theory]
    [InlineData(1024, 768, 2048, 0.03f, 1f, false)]
    [InlineData(1000, 384, 1536, 0.03f, 1f, false)]
    [InlineData(197, 768, 2304, 0.03f, 1f, false)]
    [InlineData(300, 96, 200, 1f, 1f, false)]
    [InlineData(1024, 768, 1024, 1e-7f, 1e6f, false)]
    [InlineData(1024, 768, 1024, 1e3f, 1e-6f, false)]
    [InlineData(1024, 768, 1024, 0.03f, 1f, true)]
    public void FastPrecisionErrorLikeTf32(int rows, int k, int n, float weightScale, float activationScale, bool wideRange)
    {
        Gpu.Require();
        var x = Rnd.F((long)rows * k, 1, activationScale);
        var w = Rnd.F((long)n * k, 2, weightScale);
        if (wideRange)
        {
            var rng = new Random(3);
            for (long i = 0; i < x.Length; i++)
            {
                x[i] = MathF.ScaleB(x[i], rng.Next(-20, 21));
            }
        }

        var expected = Reference(x, w, rows, k, n);
        var fastResult = Run(x, w, rows, k, n, split: true, reduced: false, MatrixPrecision.Fast);
        double fast = RelativeRms(fastResult, expected);
        double tf32 = RelativeRms(Tf32Reference(x, w, rows, k, n), expected);
        TestContext.Current.TestOutputHelper?.WriteLine($"[{rows} x {k}] x [{k} -> {n}] w~{weightScale} x~{activationScale}{(wideRange ? " wide" : "")}: relative rms fast {fast:E2}, TF32 inputs {tf32:E2}");
        Assert.All(fastResult, v => Assert.True(float.IsFinite(v)));
        Assert.True(fast < 5e-4, $"fast {fast:E2}");
        Assert.True(fast <= tf32 * 1.5, $"fast {fast:E2} vs TF32 inputs {tf32:E2}");
    }

    private static float[] Run(float[] x, float[] w, int rows, int k, int n, bool split, bool reduced, MatrixPrecision precision = MatrixPrecision.Strict)
    {
        var tg = new TG();
        var xi = tg.InF("x", TG.T, k);
        var wi = tg.Const("w", w, n, k);
        tg.B.Output("y", tg.B.MatMul(xi, wi));
        using var model = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Gpu.Device! }],
            Compile = new CompileOptions { MaxTokens = rows, AllowReducedPrecision = reduced, SplitHalfMatMul = split, MatrixPrecision = precision },
            ReservePerDevice = 0,
        });
        model.SetInput(xi, MemoryMarshal.AsBytes(x.AsSpan()));
        var p = new RunParameters();
        p.Symbols["T"] = rows;
        model.Run(p);
        var y = new float[(long)rows * n];
        model.GetOutput<float>("y", y);
        return y;
    }

    private static double[] Reference(float[] x, float[] w, int rows, int k, int n)
    {
        var y = new double[(long)rows * n];
        Parallel.For(0, rows, t =>
        {
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < k; i++)
                {
                    s += (double)x[(long)t * k + i] * w[(long)j * k + i];
                }

                y[(long)t * n + j] = s;
            }
        });
        return y;
    }

    /// <summary>The f64 product of x and w with both inputs rounded to TF32 (10 explicit mantissa bits, nearest even).</summary>
    private static float[] Tf32Reference(float[] x, float[] w, int rows, int k, int n)
    {
        static float Tf32(float v)
        {
            uint bits = BitConverter.SingleToUInt32Bits(v);
            return BitConverter.UInt32BitsToSingle((bits + 0xFFFu + ((bits >> 13) & 1u)) & 0xFFFFE000u);
        }

        var exact = Reference([.. x.Select(Tf32)], [.. w.Select(Tf32)], rows, k, n);
        return [.. exact.Select(v => (float)v)];
    }

    private static double RelativeRms(float[] got, double[] expected)
    {
        double diff = 0;
        double norm = 0;
        for (long i = 0; i < got.Length; i++)
        {
            diff += (got[i] - expected[i]) * (got[i] - expected[i]);
            norm += expected[i] * expected[i];
        }

        return Math.Sqrt(diff / norm);
    }
}
