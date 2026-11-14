using System.Runtime.InteropServices;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The split-f16 tensor-core GEMM (<see cref="CompileOptions.SplitHalfMatMul"/>) on f32 weights: its error against an f64
/// reference sits close to the f32 SIMT GEMM's and far below single f16 products (reduced precision).
/// </summary>
public sealed class SplitHalfGemmTests
{
    /// <summary>Encoder-like shapes: weights of magnitude ~0.03 (lo parts in the f16 subnormal range), activations ~1.</summary>
    [Theory]
    [InlineData(1024, 768, 2048)]
    [InlineData(1000, 384, 1536)]
    [InlineData(1576, 3072, 768)]
    public void ErrorBetweenF32AndHalf(int rows, int k, int n)
    {
        Gpu.Require();
        var x = Rnd.F((long)rows * k, 1);
        var w = Rnd.F((long)n * k, 2, 0.03f);
        var expected = Reference(x, w, rows, k, n);
        double strict = RelativeRms(Run(x, w, rows, k, n, split: false, reduced: false), expected);
        double split = RelativeRms(Run(x, w, rows, k, n, split: true, reduced: false), expected);
        double reduced = RelativeRms(Run(x, w, rows, k, n, split: false, reduced: true), expected);
        TestContext.Current.TestOutputHelper?.WriteLine($"[{rows} x {k}] x [{k} -> {n}]: relative rms strict {strict:E2}, split-f16 {split:E2}, f16 {reduced:E2}");
        Assert.True(strict < 1e-6, $"strict {strict:E2}");
        Assert.True(split < 1e-5, $"split-f16 {split:E2}");
        Assert.True(split * 20 < reduced, $"split-f16 {split:E2} vs f16 {reduced:E2}");
    }

    private static float[] Run(float[] x, float[] w, int rows, int k, int n, bool split, bool reduced)
    {
        var tg = new TG();
        var xi = tg.InF("x", TG.T, k);
        var wi = tg.Const("w", w, n, k);
        tg.B.Output("y", tg.B.MatMul(xi, wi));
        using var model = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Gpu.Vulkan! }],
            Compile = new CompileOptions { MaxTokens = rows, AllowReducedPrecision = reduced, SplitHalfMatMul = split },
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
