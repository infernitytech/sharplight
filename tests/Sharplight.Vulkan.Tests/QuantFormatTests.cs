using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The Q2_K/Q3_K, IQ1/IQ2/IQ3, TQ1_0/TQ2_0, MXFP4 and NVFP4 weight formats on the GPU against the CPU oracle, whose dequantizers
/// match ggml bit for bit (Sharplight.Cpu.Tests.GgmlDequantTests): matvec (decode), GEMM (prefill), gather and MatMulId (MoE experts).
/// </summary>
public sealed class QuantFormatTests
{
    public static IEnumerable<object[]> Formats() =>
        new[] { "q2_k", "q3_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m", "tq1_0", "tq2_0", "mxfp4", "nvfp4" }
            .Select(f => new object[] { f });

    [Theory, MemberData(nameof(Formats))]
    public void MatVec(string type) => MatMul(type, 1);

    [Theory, MemberData(nameof(Formats))]
    public void MatMulFewTokens(string type) => MatMul(type, 3);

    [Theory, MemberData(nameof(Formats))]
    public void Gemm(string type) => MatMul(type, 40);

    private static void MatMul(string type, int tokens)
    {
        Gpu.Require();
        int K = 768;
        int N = 37;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(type), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        bool tensorCores = tokens > 16;   // the prefill GEMM may round activations and weights to f16
        Diff.Check("MatMul." + type, tg.B.Build(), new Step(tokens, 0, (x, Rnd.F((long)tokens * K, 3))), atol: tensorCores ? 5e-2 : 2e-3, rtol: tensorCores ? 1e-2 : 1e-3);
    }

    [Theory, MemberData(nameof(Formats))]
    public void Gather(string type)
    {
        Gpu.Require();
        int D = 512;
        int V = 11;
        var tg = new TG();
        var idx = tg.In("idx", DataType.I32, TG.T);
        var tbl = tg.RandW("tbl", DataType.Get(type), 1f, V, D);
        tg.B.Output("y", tg.B.Gather(tbl, idx));
        Diff.Check("Gather." + type, tg.B.Build(), new Step(5, 0, (idx, new[] { 3, 0, 10, 10, 7 })), atol: 1e-6, rtol: 1e-6);
    }

    [Theory, MemberData(nameof(Formats))]
    public void MatMulIdFewItems(string type) => MatMulId(type, 3);

    [Theory, MemberData(nameof(Formats))]
    public void MatMulIdGrouped(string type) => MatMulId(type, 70);

    private static void MatMulId(string type, int tokens)
    {
        Gpu.Require();
        int E = 8;
        int S = 4;
        int N = 96;
        int K = 512;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, S);
        var w = tg.RandW("w", DataType.Get(type), 1f, E, N, K);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, w, ids]));
        var rng = new Random(11 + tokens);
        int[] expertIds = [.. Enumerable.Range(0, tokens * S).Select(_ => rng.Next(E))];
        bool grouped = tokens * S > 16;
        Diff.Check("MatMulId." + type, tg.B.Build(), new Step(tokens, 0, (x, Rnd.F((long)tokens * K, 3)), (ids, expertIds)),
            atol: grouped ? 5e-2 : 2e-3, rtol: grouped ? 1e-2 : 1e-3, maxTokens: 128);
    }
}
