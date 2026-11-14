using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The fp8 tensor-core GEMM (<c>gemm_fp8.slang</c>, prefill batches of 65+ rows with block-scaled E4M3 weights, needs
/// <c>matrix_fp8</c>) against the CPU oracle, and the f16 tensor-core GEMM that replaces it when the feature is disabled.
/// </summary>
public sealed class Fp8GemmTests
{
    private static readonly string[] s_formats = ["f8e4m3_b128", "f8e4m3_b32"];

    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in s_formats)
        {
            foreach (var (t, n) in new[] { (17, 96), (40, 100), (65, 96), (200, 100), (300, 520) })
            {
                yield return [fmt, t, n];
            }
        }
    }

    public static IEnumerable<object[]> Formats() => s_formats.Select(f => new object[] { f });

    private static void Run(string fmt, int t, int n, int k, float[] x, double atol, double rtol, IReadOnlySet<string>? disabled = null)
    {
        Gpu.Require();
        var tg = new TG();
        var xv = tg.InF("x", TG.T, k);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, n, k);
        tg.B.Output("y", tg.B.MatMul(xv, w));
        Diff.Check("MatMul.fp8." + fmt, tg.B.Build(), new Step(t, 0, (xv, x)), atol: atol, rtol: rtol, maxTokens: Math.Max(t, 64),
            what: $"T={t} N={n} K={k}", disabledFeatures: disabled);
    }

    /// <summary>Random activations: E4M3 keeps 4 significant bits, so each product errs by up to ~3 %.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void RandomActivations(string fmt, int t, int n)
    {
        const int K = 512;
        Run(fmt, t, n, K, Rnd.F((long)t * K, 3), atol: 1.5, rtol: 5e-2);
    }

    /// <summary>Activations that per-128 E4M3 quantization represents exactly: the fp8 GEMM must then match the f32 reference.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void ExactActivations(string fmt, int t, int n)
    {
        const int K = 1024;
        Run(fmt, t, n, K, Rnd.ExactFp8((long)t * K, 5), atol: 2e-3, rtol: 1e-3);
    }

    /// <summary>A grid large enough to run without split-K (and the 128-token tile with partial row and column tiles).</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void LargeGrid(string fmt)
    {
        const int K = 256;
        const int T = 1000;
        Run(fmt, T, 3000, K, Rnd.ExactFp8((long)T * K, 7), atol: 2e-3, rtol: 1e-3);
    }

    /// <summary>With <c>matrix_fp8</c> disabled the same batches run on the f16 tensor-core GEMM.</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void Fp8MatrixDisabled(string fmt)
    {
        const int K = 512;
        const int T = 130;
        Run(fmt, T, 96, K, Rnd.F((long)T * K, 3), atol: 0.25, rtol: 1e-2, disabled: new HashSet<string> { DeviceFeatures.MatrixFp8 });
    }
}
