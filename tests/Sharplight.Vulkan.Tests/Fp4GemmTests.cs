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
    /// rounding (the other paths' f16 tensor-core products of NVFP4_GS weights to about 0.5 %). K = 1,280 has an odd number of 256-value steps (NVFP4 steps then start on either half of a 32-byte sector).
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
        Diff.Check("Fp4Gemm.exact." + fmt, tg.B.Build(), new Step(t, 0, (x, ExactFp4(t, K, 7))), atol: Atol(fmt), rtol: Rtol(fmt), maxTokens: 300,
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
        // The product of two outputs: their errors on the other paths' NVFP4_GS products multiply (up to 1.6 at outputs of 30).
        Diff.Check("Fp4Gemm.dual." + fmt, tg.B.Build(), new Step(t, 0, (x, ExactFp4(t, K, 9))), atol: Atol(fmt) * 6, rtol: Rtol(fmt), maxTokens: 300,
            what: $"T={t}");
    }

    /// <summary>
    /// Three products of the same activations (a q / k / v group): on CUDA with <c>block_scaled_fp4</c> one multi-output FP4 launch over
    /// the concatenated weight rows, each block in one member's matrix; exact activations, so every output matches the oracle.
    /// </summary>
    [Theory]
    [InlineData("nvfp4_gs", 16)]
    [InlineData("nvfp4", 64)]
    [InlineData("nvfp4_gs", 200)]
    public void SharedActivations(string fmt, int t)
    {
        const int K = 768;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var type = DataType.Get(fmt);
        tg.B.Output("q", tg.B.MatMul(x, tg.RandW("wq", type, 1f, 512, K)));
        tg.B.Output("k", tg.B.MatMul(x, tg.RandW("wk", type, 1f, 128, K)));
        tg.B.Output("v", tg.B.MatMul(x, tg.RandW("wv", type, 1f, 128, K)));
        Diff.Check("Fp4Gemm.shared." + fmt, tg.B.Build(), new Step(t, 0, (x, ExactFp4(t, K, 13))), atol: Atol(fmt), rtol: Rtol(fmt), maxTokens: 256,
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
    /// Tolerances of the exact-activation checks: tight for NVFP4, the f16 products of NVFP4_GS weights on the other paths (outputs of
    /// thousands of terms of up to 60 that cancel to a few units, off by up to 0.3).
    /// </summary>
    private static double Atol(string fmt) => fmt == "nvfp4_gs" ? 5e-1 : 2e-3;

    /// <inheritdoc cref="Atol"/>
    private static double Rtol(string fmt) => fmt == "nvfp4_gs" ? 1e-2 : 4e-3;

    /// <summary>
    /// A routed expert MLP of NVFP4 expert stacks (gate / up with SiLU · mul, then down per slot, summed over the slots) at batched decode
    /// and prompt sizes: on CUDA with <c>block_scaled_fp4</c> the grouped FP4 GEMM (the gate / up pair as one dual launch), whose FP4
    /// activations put the result about 10 % (relative L2) from the oracle; a wrong routing tile, expert offset or epilogue is of order 1.
    /// </summary>
    [Theory]
    [InlineData("nvfp4_gs", 12)]
    [InlineData("nvfp4_gs", 70)]
    [InlineData("nvfp4", 200)]
    public void RoutedExperts(string fmt, int t)
    {
        Gpu.Require();
        const int K = 512;
        const int N = 256;
        const int Experts = 8;
        const int Slots = 2;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, Slots);
        var type = DataType.Get(fmt);
        var gate = tg.B.Emit1("MatMulId", [x, tg.RandW("wg", type, 0.2f, Experts, N, K), ids]);
        var up = tg.B.Emit1("MatMulId", [x, tg.RandW("wu", type, 0.2f, Experts, N, K), ids]);
        var h = tg.B.Mul(tg.B.Silu(gate), up);
        var down = tg.B.Emit1("MatMulId", [h, tg.RandW("wd", type, 0.2f, Experts, K, N), ids]);
        tg.B.Output("y", tg.B.Reshape(tg.B.Emit1("ReduceSum", [tg.B.Permute(down, 0, 2, 1)]), 0, K));
        var graph = tg.B.Build();
        var routes = new int[t * Slots];
        for (int r = 0; r < t; r++)
        {
            routes[r * Slots] = r % Experts;
            routes[r * Slots + 1] = (r * 3 + 1) % Experts == r % Experts ? (r + 1) % Experts : (r * 3 + 1) % Experts;
        }

        Step[] steps = [new Step(t, 0, (x, Rnd.F((long)t * K, 5)), (ids, routes))];
        var gpu = Diff.RunVulkan(graph, steps, maxTokens: 256)[0];
        var cpu = Diff.RunCpu(graph, steps, maxTokens: 256)[0];
        double error = 0, norm = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            error += (gpu[i] - (double)cpu[i]) * (gpu[i] - cpu[i]);
            norm += (double)cpu[i] * cpu[i];
        }

        double relative = Math.Sqrt(error / norm);
        Assert.True(relative < 0.2, $"{fmt}, T={t}: {relative:E2} (relative L2) away from the oracle");
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
