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
    /// <summary>
    /// Devices without int8 matrices run these batches on a GEMM with f16 operands, which the exact-activation tolerances do not
    /// allow; only the 8x8 form (Metal) is turned off, so devices with int8 matrices still run the int8 GEMM under test.
    /// </summary>
    private static readonly HashSet<string> s_noHalfOperands = [DeviceFeatures.MatrixF16Small];

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
        => WithEveryFormat(() =>
        {
            var tg = new TG();
            var xv = tg.InF("x", TG.T, k);
            var w = tg.RandW("w", DataType.Get(fmt), 1f, n, k);
            tg.B.Output("y", tg.B.MatMul(xv, w));
            Diff.Check("MatMul.mmq." + fmt, tg.B.Build(), new Step(t, 0, (xv, x)), atol: atol, rtol: rtol, maxTokens: Math.Max(t, 64),
                what: $"T={t} N={n} K={k}", disabledFeatures: disabled);
        });

    /// <summary>Runs <paramref name="test"/> with every supported weight format on the int8 GEMM at every size.</summary>
    private static void WithEveryFormat(Action test)
    {
        Gpu.Require();
        var old = GraphLowerer.MmqFormats;
        var oldLargeOnly = GraphLowerer.MmqLargeOnlyFormats;
        GraphLowerer.MmqFormats = GraphLowerer.MmqSupportedFormats;
        GraphLowerer.MmqLargeOnlyFormats = new HashSet<DataType>();
        try
        {
            test();
        }
        finally
        {
            GraphLowerer.MmqFormats = old;
            GraphLowerer.MmqLargeOnlyFormats = oldLargeOnly;
        }
    }

    /// <summary>
    /// A row norm read by Q8_0 matmuls writes the q8_1 rows of their int-dot matvecs for decode steps; at prefill sizes the int8
    /// GEMM readers (from 33 rows) must quantize into those rows themselves. #78: when no reader's int-dot range extended the copy
    /// past 8 rows, nothing did, and the GEMMs read unwritten rows (Qwen3.8-27B Q8_0: NaN logits from 171-token prompts).
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(40)]
    [InlineData(200)]
    public void NormThenGemms(int t) => WithEveryFormat(() =>
    {
        const int K = 1024;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var n = tg.B.RmsNorm(x, tg.RandW("nw", DataType.F32, 1f, K), 1e-6f);
        var g = tg.B.MatMul(n, tg.RandW("wg", DataType.Q8_0, 0.05f, 1024, K));
        var u = tg.B.MatMul(n, tg.RandW("wu", DataType.Q8_0, 0.05f, 1024, K));
        tg.B.Output("h", tg.B.Mul(tg.B.Silu(g), u));
        Diff.Check("NormThenMmq", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: 2e-2, rtol: 2e-2, maxTokens: 256,
            what: $"T={t}");
    });

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
        Run(fmt, t, n, K, Rnd.ExactInt8((long)t * K, 5), atol: 2e-3, rtol: 1e-3, disabled: s_noHalfOperands);
    }

    /// <summary>A grid large enough to run without split-K (and the 128-token tile with partial row and column tiles).</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void LargeGrid(string fmt)
    {
        const int K = 256;
        const int T = 1000;
        Run(fmt, T, 3000, K, Rnd.ExactInt8((long)T * K, 7), atol: 2e-3, rtol: 1e-3, disabled: s_noHalfOperands);
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
