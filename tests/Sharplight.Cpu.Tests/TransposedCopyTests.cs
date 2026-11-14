namespace Sharplight.Cpu.Tests;

/// <summary>Copies out of transposed views (the blocked transpose path of the strided copy), e.g. a ViT's class-token concat.</summary>
public class TransposedCopyTests
{
    [Fact]
    public void ConcatOfTransposedView()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 37, 21);                  // [N, 37, 21]
        var cls = tg.InF("cls", TestGraph.T, 1, 37);               // [N, 1, 37]
        var t = tg.B.Permute(x, 0, 2, 1);                          // [N, 21, 37], source unit stride on axis 1
        tg.B.Output("y", tg.B.Concat(1, cls, t));                  // [N, 22, 37]
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        var xd = Kit.Rand(2 * 37 * 21, 3);
        var cd = Kit.Rand(2 * 37, 4);
        Kit.Set(m, x, xd);
        Kit.Set(m, cls, cd);
        var p = Kit.P(2);
        m.Run(p);
        var expect = new float[2 * 22 * 37];
        for (int n = 0; n < 2; n++)
        {
            for (int c = 0; c < 37; c++)
            {
                expect[(n * 22) * 37 + c] = cd[n * 37 + c];
            }

            for (int r = 0; r < 21; r++)
            {
                for (int c = 0; c < 37; c++)
                {
                    expect[(n * 22 + 1 + r) * 37 + c] = xd[(n * 37 + c) * 21 + r];
                }
            }
        }

        Kit.AssertClose(expect, Kit.Out(m, "y", p), 0, 0);
    }

    [Fact]
    public void ContiguousOfRank4Permute()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 19, 5, 18);               // [N, C, H, W]
        tg.B.Output("y", tg.B.Contiguous(tg.B.Permute(x, 0, 2, 3, 1)));   // [N, H, W, C]
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        var xd = Kit.Rand(2 * 19 * 5 * 18, 5);
        Kit.Set(m, x, xd);
        var p = Kit.P(2);
        m.Run(p);
        var expect = new float[xd.Length];
        for (int n = 0; n < 2; n++)
        {
            for (int c = 0; c < 19; c++)
            {
                for (int h = 0; h < 5; h++)
                {
                    for (int w = 0; w < 18; w++)
                    {
                        expect[((n * 5 + h) * 18 + w) * 19 + c] = xd[((n * 19 + c) * 5 + h) * 18 + w];
                    }
                }
            }
        }

        Kit.AssertClose(expect, Kit.Out(m, "y", p), 0, 0);
    }
}
