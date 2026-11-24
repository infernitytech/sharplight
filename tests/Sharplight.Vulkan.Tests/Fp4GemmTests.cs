using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Matrix products of NVFP4 weights at the batch sizes of the CUDA provider's FP4 GEMM (<c>gemm_fp4_native.cu</c>, from 9 rows where the
/// device has <c>block_scaled_fp4</c>; other providers and devices run the integer-dot matvecs and the GEMMs on dequantized weights),
/// against the CPU oracle.
/// </summary>
public sealed class Fp4GemmTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "nvfp4", "nvfp4_gs" })
        {
            foreach (var t in new[] { 9, 16, 17, 33, 64, 65, 200, 300 })
            {
                yield return [fmt, t];
            }
        }
    }

    /// <summary>
    /// Activations FP4 represents exactly (E2M1 values times a power of two per 16-value group, each group holding its largest value
    /// and the row's first group the row's maximum 6): the FP4 GEMM's quantization keeps them, so the products match the oracle to f32
    /// rounding. K = 1,280 has an odd number of 256-value steps (NVFP4 steps then start on either half of a 32-byte sector).
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void ExactActivations(string fmt, int t)
    {
        const int K = 1280;
        const int N = 136;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 1f, N, K)));
        Diff.Check("Fp4Gemm.exact." + fmt, tg.B.Build(), new Step(t, 0, (x, ExactFp4(t, K, 7))), atol: 2e-3, rtol: 1e-3, maxTokens: 300,
            what: $"T={t}");
    }

    /// <summary>The gate / up pair with SiLU · mul in the FP4 GEMM's dual store, on exactly representable activations.</summary>
    [Theory]
    [InlineData("nvfp4", 12)]
    [InlineData("nvfp4_gs", 40)]
    [InlineData("nvfp4_gs", 130)]
    [InlineData("nvfp4", 260)]
    public void DualGateUp(string fmt, int t)
    {
        const int K = 1024;
        const int N = 192;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var gate = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.2f, N, K));
        var up = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.2f, N, K));
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(gate), up));
        Diff.Check("Fp4Gemm.dual." + fmt, tg.B.Build(), new Step(t, 0, (x, ExactFp4(t, K, 9))), atol: 2e-3, rtol: 2e-3, maxTokens: 300,
            what: $"T={t}");
    }

    /// <summary>
    /// Random activations: the FP4 GEMM quantizes them to E2M1 with an E4M3 scale per 16 values (about 9 % relative error per value,
    /// the W4A4 mode), so the products are about 9 % (relative L2) away from the oracle's; the other paths quantize activations to
    /// 8 bits or none. A wrong scale or layout gives errors of order 1.
    /// </summary>
    [Theory]
    [InlineData("nvfp4", 24)]
    [InlineData("nvfp4_gs", 64)]
    [InlineData("nvfp4_gs", 256)]
    public void RandomActivations(string fmt, int t)
    {
        Gpu.Require();
        const int K = 2048;
        const int N = 256;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 1f, N, K)));
        var graph = tg.B.Build();
        Step[] steps = [new Step(t, 0, (x, Rnd.F((long)t * K, 3)))];
        var gpu = Diff.RunVulkan(graph, steps, maxTokens: 256)[0];
        var cpu = Diff.RunCpu(graph, steps, maxTokens: 256)[0];
        double error = 0, norm = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            error += (gpu[i] - (double)cpu[i]) * (gpu[i] - cpu[i]);
            norm += (double)cpu[i] * cpu[i];
        }

        double relative = Math.Sqrt(error / norm);
        Assert.True(relative < 0.15, $"{fmt}, T={t}: {relative:E2} (relative L2) away from the oracle");
    }

    /// <summary>
    /// [<paramref name="rows"/>, <paramref name="k"/>] activations that NVFP4 quantization per 16 values keeps exactly: each group is
    /// E2M1 values times 2^-j (j from 0 to 3, 0 in the row's first group) and holds ±6 · 2^-j.
    /// </summary>
    private static float[] ExactFp4(int rows, int k, int seed)
    {
        float[] magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];
        var rng = new Random(seed);
        var data = new float[rows * k];
        for (int r = 0; r < rows; r++)
        {
            for (int g = 0; g < k / 16; g++)
            {
                float scale = MathF.ScaleB(1f, g == 0 ? 0 : -rng.Next(4));
                int peak = rng.Next(16);
                for (int i = 0; i < 16; i++)
                {
                    float m = i == peak ? 6f : magnitudes[rng.Next(8)];
                    data[r * k + g * 16 + i] = (rng.Next(2) == 0 ? m : -m) * scale;
                }
            }
        }

        return data;
    }
}
