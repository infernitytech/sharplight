using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// <see cref="Runtime.CompileOptions.RowInvariantRows"/>: a step of up to that many activation rows must compute every row
/// bit-identically to a step of that row alone, as speculative verification needs (its rows must reproduce plain decoding).
/// Matmuls over every weight format family (f32 matvec formats, the IQ codebook formats, the integer-dot formats whose single rows take the f32 matvec
/// or the integer-dot matvec, block FP8) and widths that put few-row batches on the int-dot matvec or the skinny GEMM,
/// single and as fused gate / up pairs, and expert-indexed (MoE) matmuls; rows 2-4 against single-row steps over the same rows.
/// The f32 matvec writes its sums as an explicit chain of fused multiply-adds, so its variants agree by construction. Formats
/// whose single rows take the integer-dot matvec (K-quants) rely on the driver contracting each variant's plain sums alike:
/// it does at these shapes but not at every one (q4_k [2048→48] differs in the last bit), so verification on such weights is
/// not guaranteed to be bit-exact.
/// </summary>
public sealed class RowInvarianceTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_k", "q6_k", "f8e4m3_b128", "iq1_s", "iq2_xxs", "iq3_xxs", "iq4_nl", "iq4_xs" })
        {
            foreach (var (k, n) in new[] { (1024, 4096), (512, 1024), (2048, 48) })
            {
                if (fmt != "q4_k" || n != 48)
                {
                    yield return [fmt, k, n];
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatMul(string fmt, int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 0.1f, n, k)));
        Check(tg, x, k);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DualMatMul(string fmt, int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.1f, n, k));
        var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.1f, n, k));
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
        Check(tg, x, k);
    }

    /// <summary>Expert-indexed matmuls (MoE, two experts per row chosen by a router on the row), as the routed experts of a verification step run.</summary>
    [Theory]
    [InlineData("q8_0")]
    [InlineData("q4_k")]
    [InlineData("iq1_s")]
    [InlineData("iq4_nl")]
    public void ExpertMatMul(string fmt)
    {
        const int K = 512;
        const int N = 256;
        const int Experts = 8;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var router = tg.B.MatMul(x, tg.RandW("router", DataType.F32, 0.1f, Experts, K));
        var ids = tg.B.Emit("TopK", [router], Attributes.From(("k", 2L)))[1];
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, tg.RandW("experts", DataType.Get(fmt), 0.1f, Experts, N, K), ids]));
        Check(tg, x, K);
    }

    /// <summary>Runs rows 1-4 one per step, then the same rows in one step of 2, 3 and 4 rows each, and compares bit for bit.</summary>
    private static void Check(TG tg, Value x, int k)
    {
        Gpu.Require();
        const int Rows = 4;
        var data = Rnd.F((long)Rows * k, 7);
        var graph = tg.B.Build();
        var steps = new List<Step>();
        for (int r = 0; r < Rows; r++)
        {
            steps.Add(new Step(1, 0, (x, data[(r * k)..((r + 1) * k)])));
        }

        for (int t = 2; t <= Rows; t++)
        {
            steps.Add(new Step(t, 0, (x, data[..(t * k)])));
        }

        var outputs = Diff.RunVulkan(graph, steps, maxTokens: 16, rowInvariantRows: Rows);
        var single = outputs[..Rows];
        for (int t = 2; t <= Rows; t++)
        {
            var batched = outputs[Rows + t - 2];
            int width = batched.Length / t;
            for (int r = 0; r < t; r++)
            {
                for (int i = 0; i < width; i++)
                {
                    if (BitConverter.SingleToInt32Bits(batched[r * width + i]) != BitConverter.SingleToInt32Bits(single[r][i]))
                    {
                        Assert.Fail($"{t}-row step, row {r}, column {i}: {batched[r * width + i]:R} vs {single[r][i]:R} in a single-row step");
                    }
                }
            }
        }
    }
}
