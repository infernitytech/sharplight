using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>Packed-sequence ops of embedding graphs (Attention with i32 spans, SequencePool): Vulkan vs. the CPU oracle.</summary>
public class PackedSequenceTests
{
    private static int[] RowSpans(int[] lengths)
    {
        var spans = new int[lengths.Sum() * 2];
        int first = 0;
        foreach (int len in lengths)
        {
            for (int t = first; t < first + len; t++) { spans[t * 2] = first; spans[t * 2 + 1] = len; }
            first += len;
        }
        return spans;
    }

    // Short batches take the scalar kernel, long ones the tensor-core flash kernel (32-row blocks straddling sequences).
    [Theory]
    [InlineData(false, 4, 4, 64, 64, new[] { 3, 1, 2 })]
    [InlineData(true, 4, 4, 64, 64, new[] { 3, 1, 2 })]
    [InlineData(false, 12, 12, 64, 64, new[] { 7, 40, 1, 33, 64, 19 })]
    [InlineData(true, 16, 8, 128, 128, new[] { 7, 40, 1, 33, 64, 19 })]
    [InlineData(false, 6, 2, 32, 32, new[] { 100, 3, 50 })]
    [InlineData(true, 8, 2, 64, 64, new[] { 29, 31, 2, 90 })]
    public void AttentionWithinSpans(bool causal, int Hq, int Hkv, int D, int Dv, int[] lengths)
    {
        int T = lengths.Sum();
        var tg = new TG(); var Tm = TG.T;
        var q = tg.InF("q", Tm, Hq, D); var k = tg.InF("k", Tm, Hkv, D); var v = tg.InF("v", Tm, Hkv, Dv);
        var s = tg.In("spans", DataType.I32, Tm, 2);
        tg.B.Output("o", tg.B.Emit1("Attention", [q, k, v, s], Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("causal", causal))));
        var step = new Step(T, 0, (q, Rnd.F(T * Hq * D, 1)), (k, Rnd.F(T * Hkv * D, 2)), (v, Rnd.F(T * Hkv * Dv, 3)), (s, RowSpans(lengths)));
        // The flash kernel runs Q/K/P in f16.
        Diff.Check("Attention.spans", tg.B.Build(), step, atol: 3e-3, rtol: 1e-2, maxTokens: 256, what: $"causal={causal} Hq={Hq} Hkv={Hkv} D={D} [{string.Join(",", lengths)}]");
    }

    [Theory]
    [InlineData(2), InlineData(40)]
    public void CrossAttentionWithSpans(int queries)
    {
        // Pooling-style cross attention: a few query rows each attending one span of a longer key sequence.
        int H = 4, D = 32, S = 96;
        var tg = new TG(); var Sm = TG.T; var Qm = Dim.Sym("R");
        var q = tg.InF("q", Qm, H, D); var k = tg.InF("k", Sm, H, D); var v = tg.InF("v", Sm, H, D);
        var s = tg.In("spans", DataType.I32, Qm, 2);
        tg.B.Output("o", tg.B.Emit1("Attention", [q, k, v, s], Attributes.From(("scale", 0.2f))));
        var spans = new int[queries * 2];
        for (int i = 0; i < queries; i++) { int a = (i * 13) % 60; spans[i * 2] = a; spans[i * 2 + 1] = 1 + (i * 7) % (S - a - 1); }
        var step = new Step(S, 0, (q, Rnd.F(queries * H * D, 1)), (k, Rnd.F(S * H * D, 2)), (v, Rnd.F(S * H * D, 3)), (s, spans)) { R = queries };
        Diff.Check("Attention.cross", tg.B.Build(), step, atol: 3e-3, rtol: 1e-2, maxTokens: 128, what: $"queries={queries}");
    }

    [Theory]
    [InlineData("mean"), InlineData("sum"), InlineData("max"), InlineData("first"), InlineData("last")]
    public void SequencePool(string mode)
    {
        int D = 72;
        int[] spans = [0, 3, 3, 1, 4, 0, 4, 30, 34, 5];
        var tg = new TG(); var Tm = TG.T; var Bm = Dim.Sym("R");
        var x = tg.InF("x", Tm, D); var s = tg.In("spans", DataType.I32, Bm, 2);
        tg.B.Output("y", tg.B.Emit1("SequencePool", [x, s], Attributes.From(("mode", mode))));
        var step = new Step(41, 0, (x, Rnd.F(41 * D, 9)), (s, spans)) { R = spans.Length / 2 };
        Diff.Check("SequencePool", tg.B.Build(), step, atol: 1e-5, rtol: 1e-5, what: mode);
    }
}
