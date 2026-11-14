using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Reshapes of strided views (column slices of a wider product) that merge the symbolic axis with inner axes.</summary>
public class StridedReshapeTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void SliceMergedWithSymbolicAxis(int part)
    {
        const int S = 6;
        const int H = 3;
        const int D = 4;
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, S, 2 * H * D);                         // [N, S, 2·H·D]
        var slice = tg.B.Slice(x, 2, part * H * D, H * D);                      // [N, S, H·D], row stride 2·H·D
        var merged = tg.B.Emit1("Reshape", [slice], Attributes.From(("shape", (long[])[0, H, D]), ("outer_symbol", (TestGraph.T * Dim.Of(S)).Symbol!)));
        tg.B.Output("y", tg.B.Contiguous(merged));                              // [N·S, H, D]
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        var xd = Kit.Rand(3 * S * 2 * H * D, 7);
        Kit.Set(m, x, xd);
        var p = Kit.P(2);
        m.Run(p);
        var expect = new float[2 * S * H * D];
        for (int r = 0; r < 2 * S; r++)
        {
            for (int c = 0; c < H * D; c++)
            {
                expect[r * H * D + c] = xd[r * 2 * H * D + part * H * D + c];
            }
        }

        Kit.AssertClose(expect, Kit.Out(m, "y", p), 0, 0);
    }
}
