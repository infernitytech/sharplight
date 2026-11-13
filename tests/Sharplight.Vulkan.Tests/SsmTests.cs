using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public class SsmTests
{
    // (H, P, N, G): tiny, odd sizes, Nemotron-H Nano 4B, Nemotron-3-Nano-30B.
    [Theory]
    [InlineData(4, 8, 16, 2, false)] [InlineData(4, 8, 16, 2, true)] [InlineData(3, 5, 8, 3, true)] [InlineData(2, 6, 12, 1, false)]
    [InlineData(96, 80, 128, 8, true)] [InlineData(64, 64, 128, 8, false)]
    public void SelectiveScanAcrossSteps(int H, int P, int N, int G, bool sliced)
    {
        Gpu.Require();
        int width = H * P + 2 * G * N;
        var tg = new TG();
        var xi = tg.InF("x", TG.T, sliced ? width : H * P); var dti = tg.InF("dt", TG.T, H);
        var bi = tg.InF("b", TG.T, G * N); var ci = tg.InF("c", TG.T, G * N);
        Value xv, bv, cv;
        if (sliced)
        {
            xv = tg.B.Reshape(tg.B.Slice(xi, 1, 0, H * P), 0, H, P);
            bv = tg.B.Reshape(tg.B.Slice(xi, 1, H * P, G * N), 0, G, N);
            cv = tg.B.Reshape(tg.B.Slice(xi, 1, H * P + G * N, G * N), 0, G, N);
        }
        else { xv = tg.B.Reshape(xi, 0, H, P); bv = tg.B.Reshape(bi, 0, G, N); cv = tg.B.Reshape(ci, 0, G, N); }
        var dtb = tg.RandW("dtb", DataType.F32, 1f, H);
        var A = tg.RandW("A", DataType.F32, 1f, H);
        var Dw = tg.RandW("D", DataType.F32, 1f, H);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, H, P, N);
        // A must be negative: A' = -(|A|+0.1) computed in the graph.
        var Aneg = tg.B.Scale(tg.B.Unary("Abs", A), -1f, -0.1f);
        tg.B.Output("y", tg.B.Emit("SelectiveScan", [xv, dti, dtb, Aneg, bv, cv, Dw, st], Attributes.From(("dt_min", 0.001f)), writes: [st])[0]);
        int[] chunks = [3, 1, 2, 1, 9];
        var steps = chunks.Select((c, i) => new Step(c, 0,
            (xi, Rnd.F((long)c * (sliced ? width : H * P), 1 + i, 0.5f)), (dti, Rnd.F(c * H, 11 + i, 2f)),
            (bi, Rnd.F((long)c * G * N, 21 + i, 0.5f)), (ci, Rnd.F((long)c * G * N, 31 + i, 0.5f)))).ToList();
        Diff.Check("SelectiveScan", tg.B.Build(), steps, atol: 2e-4, rtol: 2e-3, maxTokens: 16, what: $"H={H} P={P} N={N} G={G} sliced={sliced}");
    }

    [Theory]
    [InlineData(20, 4, true)] [InlineData(600, 4, false)] [InlineData(33, 2, true)]
    public void CausalConvBiasAcrossSteps(int C, int K, bool silu)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, C);
        var w = tg.RandW("w", DataType.F32, 1f, C, K);
        var bias = tg.RandW("bias", DataType.F32, 1f, C);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [x, w, state, bias], Attributes.From(("activation", silu ? "silu" : "none"), ("bias", true)), writes: [state])[0]);
        int[] chunks = [3, 1, 5, 2];
        var steps = chunks.Select((c, i) => new Step(c, 0, (x, Rnd.F((long)c * C, 100 + i)))).ToList();
        Diff.Check("CausalConv1d.bias", tg.B.Build(), steps, atol: 2e-5, rtol: 1e-4, maxTokens: 8, what: $"C={C} K={K} silu={silu}");
    }
}
