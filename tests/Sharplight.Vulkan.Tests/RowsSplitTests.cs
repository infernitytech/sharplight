using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Softmax and GroupNorm over rows long enough for the chunked two-pass kernels (rows_split.slang): partial statistics
/// per chunk, combined per row, including a partial last chunk and several rows per token.
/// </summary>
public sealed class RowsSplitTests
{
    [Theory]
    [InlineData(1, 1, 50000, 1f, false)]
    [InlineData(2, 3, 20001, 0.5f, false)]
    [InlineData(2, 1, 40000, 1f, true)]
    public void Softmax(int t, int r1, int d, float scale, bool log)
    {
        Gpu.Require();
        var tg = new TG();
        var x = r1 == 1 ? tg.InF("x", TG.T, d) : tg.InF("x", TG.T, r1, d);
        tg.B.Output("y", tg.B.Emit1("Softmax", [x], Attributes.From(("scale", scale), ("log", log))));
        Diff.Check("Softmax.split", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * r1 * d, 7, 8f))), atol: 1e-7, rtol: 1e-4, maxTokens: 2,
            what: $"r1={r1} d={d} log={log}");
    }

    [Theory]
    [InlineData(32, 4, 70, true, true)]
    [InlineData(16, 16, 130, false, false)]
    [InlineData(12, 3, 75, true, false)]
    public void GroupNorm(int c, int groups, int hw, bool scale, bool bias)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        var args = new List<Value> { x };
        if (scale)
        {
            args.Add(tg.RandW("s", DataType.F32, 1f, c));
        }

        if (bias)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, c));
        }

        tg.B.Output("y", tg.B.Emit1("GroupNorm", args, Attributes.From(("groups", (long)groups), ("eps", 1e-5f))));
        Diff.Check("GroupNorm.split", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2L * c * hw * hw, 8, 3f))), atol: 5e-5, rtol: 1e-4, maxTokens: 2,
            what: $"c={c} g={groups} {hw}x{hw}");
    }
}
