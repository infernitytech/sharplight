using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The int8 tensor-core GEMM (<c>mmq.slang</c>, prefill batches of 65+ rows, needs <c>matrix_int8</c>) against the CPU oracle,
/// and the f16 tensor-core GEMM that replaces it when the feature is disabled. Every supported weight format is switched to
/// the int8 GEMM for these tests (by default only some formats use it).
/// </summary>
[Collection("Mmq")]
public sealed class MmqTests
{
    private static readonly string[] s_formats = ["q8_0", "q4_k", "q5_k", "q6_k", "iq4_nl", "iq4_xs"];

    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in s_formats)
        {
            foreach (var (t, n) in new[] { (65, 96), (200, 100), (300, 520) })
            {
                yield return [fmt, t, n];
            }
        }
    }

    public static IEnumerable<object[]> Formats() => s_formats.Select(f => new object[] { f });

    private static void Run(string fmt, int t, int n, int k, float[] x, double atol, double rtol, IReadOnlySet<string>? disabled = null)
    {
        Gpu.Require();
        var old = GraphLowerer.MmqFormats;
        var oldLargeOnly = GraphLowerer.MmqLargeOnlyFormats;
        GraphLowerer.MmqFormats = GraphLowerer.MmqSupportedFormats;
        GraphLowerer.MmqLargeOnlyFormats = new HashSet<DataType>();
        try
        {
            var tg = new TG();
            var xv = tg.InF("x", TG.T, k);
            var w = tg.RandW("w", DataType.Get(fmt), 1f, n, k);
            tg.B.Output("y", tg.B.MatMul(xv, w));
            Diff.Check("MatMul.mmq." + fmt, tg.B.Build(), new Step(t, 0, (xv, x)), atol: atol, rtol: rtol, maxTokens: Math.Max(t, 64),
                what: $"T={t} N={n} K={k}", disabledFeatures: disabled);
        }
        finally
        {
            GraphLowerer.MmqFormats = old;
            GraphLowerer.MmqLargeOnlyFormats = oldLargeOnly;
        }
    }

    /// <summary>
    /// Random activations: 8-bit block quantization errs in proportion to the weight row's norm (large for these random
    /// weights), as for the integer-dot matvec in <see cref="MatMulTests"/>.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void RandomActivations(string fmt, int t, int n)
    {
        const int K = 512;
        Run(fmt, t, n, K, Rnd.F((long)t * K, 3), atol: 3.0, rtol: 1e-2);
    }

    /// <summary>Activations that q8_1 represents exactly: the int8 GEMM must then match the f32 reference closely.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void ExactActivations(string fmt, int t, int n)
    {
        const int K = 1024;
        Run(fmt, t, n, K, Rnd.ExactInt8((long)t * K, 5), atol: 2e-3, rtol: 1e-3);
    }

    /// <summary>A grid large enough to run without split-K (and the 128-token tile with partial row and column tiles).</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void LargeGrid(string fmt)
    {
        const int K = 256;
        const int T = 1000;
        Run(fmt, T, 3000, K, Rnd.ExactInt8((long)T * K, 7), atol: 2e-3, rtol: 1e-3);
    }

    /// <summary>With <c>matrix_int8</c> disabled the same batches run on the f16 tensor-core GEMM.</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void Int8MatrixDisabled(string fmt)
    {
        const int K = 512;
        const int T = 130;
        Run(fmt, T, 96, K, Rnd.F((long)T * K, 3), atol: 0.25, rtol: 1e-2, disabled: new HashSet<string> { DeviceFeatures.MatrixInt8 });
    }
}
