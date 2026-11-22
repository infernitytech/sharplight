using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The K-quant decode GEMM at decode batches of 9-64 rows against the CPU: on Vulkan devices with the MMA fragment layout the
/// cooperative-matrix kernel (<c>gemm_kq.slang</c>), on CUDA the hand-written one (<c>gemm_kq_native.cu</c>), elsewhere the generated
/// GEMMs. Single projections with bias rows (a partial last weight tile; K = 1280: five superblocks, Q6_K rows of 1050 bytes whose
/// blocks start at any even byte), the q / k / v multi-output form, and the dual gate / up form with its activation in the store.
/// </summary>
public sealed class KQuantDecodeGemmTests
{
    private static DataType Format(string format) => format switch { "q4_k" => DataType.Q4_K, "q5_k" => DataType.Q5_K, _ => DataType.Q6_K };

    [Theory]
    [InlineData("q4_k", 9)]
    [InlineData("q4_k", 16)]
    [InlineData("q4_k", 32)]
    [InlineData("q4_k", 47)]
    [InlineData("q4_k", 64)]
    [InlineData("q5_k", 17)]
    [InlineData("q5_k", 40)]
    [InlineData("q5_k", 64)]
    [InlineData("q6_k", 12)]
    [InlineData("q6_k", 33)]
    [InlineData("q6_k", 64)]
    public void ProjectionWithBias(string format, int rows)
    {
        Gpu.Require();
        const int K = 1280, N = 336;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var y = tg.B.MatMul(x, tg.RandW("w", Format(format), 0.05f, N, K));
        tg.B.Output("y", tg.B.Add(y, tg.RandW("b", DataType.F32, 0.5f, N)));
        Diff.Check("KQuantDecode." + format, tg.B.Build(), new Step(rows, 0, (x, Rnd.F((long)rows * K, 3))), atol: 2e-2, rtol: 1e-2, what: $"T={rows}");
    }

    /// <summary>Wide matrices, whose grids of many 64-row weight tiles split K between workgroups or not depending on the batch.</summary>
    [Theory]
    [InlineData("q4_k", 24, 4096, 4096)]
    [InlineData("q4_k", 64, 12288, 1024)]
    [InlineData("q6_k", 48, 1024, 3072)]
    [InlineData("q6_k", 20, 4096, 1536)]
    public void WideProjection(string format, int rows, int n, int k)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", Format(format), 0.05f, n, k)));
        Diff.Check("KQuantDecode.wide." + format, tg.B.Build(), new Step(rows, 0, (x, Rnd.F((long)rows * k, 5))), atol: 3e-2, rtol: 1e-2,
            what: $"[{k}→{n}] T={rows}");
    }

    [Theory]
    [InlineData("q4_k", 40)]
    [InlineData("q5_k", 64)]
    [InlineData("q6_k", 20)]
    public void QkvProjections(string format, int rows)
    {
        Gpu.Require();
        const int K = 1280;
        int[] widths = [256, 128, 80];
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        for (int o = 0; o < widths.Length; o++)
        {
            var y = tg.B.MatMul(x, tg.RandW("w" + o, Format(format), 0.05f, widths[o], K));
            tg.B.Output("y" + o, tg.B.Add(y, tg.RandW("b" + o, DataType.F32, 0.5f, widths[o])));
        }

        Diff.Check("KQuantDecode.qkv." + format, tg.B.Build(), new Step(rows, 0, (x, Rnd.F((long)rows * K, 7))), atol: 2e-2, rtol: 1e-2, what: $"T={rows}");
    }

    /// <summary>silu(x · Wgᵀ) · (x · Wuᵀ) (or gelu_tanh); 1000 outputs end in a partial 32-output tile of the dual form.</summary>
    [Theory]
    [InlineData("q4_k", 9, false, 4096)]
    [InlineData("q4_k", 16, false, 4096)]
    [InlineData("q4_k", 40, false, 4096)]
    [InlineData("q4_k", 64, false, 4096)]
    [InlineData("q6_k", 20, true, 1000)]
    [InlineData("q6_k", 48, false, 4096)]
    [InlineData("q4_k", 20, true, 1000)]
    [InlineData("q5_k", 64, true, 1000)]
    public void GatedProjection(string format, int rows, bool gelu, int n)
    {
        Gpu.Require();
        const int K = 512;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var gate = tg.B.MatMul(x, tg.RandW("wg", Format(format), 0.05f, n, K));
        var y = tg.B.Mul(gelu ? tg.B.Unary("GeluTanh", gate) : tg.B.Silu(gate), tg.B.MatMul(x, tg.RandW("wu", Format(format), 0.05f, n, K)));
        tg.B.Output("y", y);
        // silu(g) · u multiplies the errors of both products (f16 dequantized weights and activations), which are proportional to the
        // weight rows' norms rather than to each output: the bound is relative to the largest output, as on CUDA.
        var graph = tg.B.Build();
        Step[] steps = [new Step(rows, 0, (x, Rnd.F((long)rows * K, 11)))];
        var actual = Diff.RunVulkan(graph, steps)[0];
        var expected = Diff.RunCpu(graph, steps)[0];
        double scale = expected.Max(MathF.Abs);
        Assert.True(scale > 0);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 3e-3 * scale, $"T={rows} N={n} y[{i}]: {actual[i]} vs {expected[i]} (scale {scale})");
        }
    }

    /// <summary>
    /// The same projection with the K-quant decode GEMMs and with the generated gemm_cm2 (<see cref="DeviceFeatures.MatrixMmaLayout"/>
    /// and <see cref="DeviceFeatures.NativeGemmKQuant"/> disabled) agree closely: both dequantize the weights to f16 and read the f16
    /// copy of x.
    /// </summary>
    [Theory]
    [InlineData("q4_k", 32, false)]
    [InlineData("q6_k", 64, false)]
    [InlineData("q6_k", 20, true)]
    [InlineData("q5_k", 48, true)]
    public void MatchesTheGeneratedGemm(string format, int rows, bool gated)
    {
        Gpu.Require();
        const int K = 1280, N = 336;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var y = tg.B.MatMul(x, tg.RandW("w", Format(format), 0.05f, N, K));
        if (gated)
        {
            y = tg.B.Mul(tg.B.Silu(y), tg.B.MatMul(x, tg.RandW("u", Format(format), 0.05f, N, K)));
        }

        tg.B.Output("y", y);
        var graph = tg.B.Build();
        Step[] steps = [new Step(rows, 0, (x, Rnd.F((long)rows * K, 13)))];
        var actual = Diff.RunVulkan(graph, steps)[0];
        var reference = Diff.RunVulkan(graph, steps, disabledFeatures: new HashSet<string> { DeviceFeatures.MatrixMmaLayout, DeviceFeatures.NativeGemmKQuant })[0];
        double scale = reference.Max(MathF.Abs);
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(Math.Abs(reference[i] - actual[i]) <= 1e-3 * scale, $"y[{i}]: {actual[i]} vs the generated GEMM's {reference[i]}");
        }
    }
}
