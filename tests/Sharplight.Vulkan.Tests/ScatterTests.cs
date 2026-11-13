using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// ScatterND lowerings against the CPU oracle: chains with constant indices folded into gather kernels (the exported
/// "fold" pattern), chains split over several passes, chains broken by escaping intermediates, and the parallel kernel
/// for runtime indices.
/// </summary>
public sealed class ScatterTests
{
    private static Value IndexConstant(TG tg, string name, int[] values, params long[] dims)
        => tg.B.Constant(name, DataType.I32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    /// <summary>
    /// Patch folding as exported by MobileViT: 16 scatters each place one 2x2 patch (a slice of the same source) into an
    /// [2, 4, 8, 8] map, with dead slices of every intermediate left behind.
    /// </summary>
    [Fact]
    public void FoldChain()
    {
        var tg = new TG();
        var x = tg.InF("x", 2, 4, 16, 4);
        var y = tg.WF("zeros", new float[2 * 4 * 8 * 8], 2, 4, 8, 8);
        for (int p = 0; p < 16; p++)
        {
            int ph = p / 4;
            int pw = p % 4;
            var slice = tg.B.Emit1("Slice", [x], Attributes.From(("axis", 2L), ("start", (long)p), ("length", 1L)));
            var update = tg.B.Emit1("Reshape", [slice], Attributes.From(("shape", new long[] { 2, 4, 2, 2 })));
            var indices = new List<int>();
            for (int n = 0; n < 2; n++)
            {
                for (int c = 0; c < 4; c++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            indices.AddRange([n, c, 2 * ph + i, 2 * pw + j]);
                        }
                    }
                }
            }

            y = tg.B.Emit1("ScatterND", [y, IndexConstant(tg, $"idx{p}", [.. indices], 2, 4, 2, 2, 4), update]);
            tg.B.Emit1("Slice", [y], Attributes.From(("axis", 2L), ("start", 0L), ("length", 2L)));   // dead
        }

        tg.B.Output("y", y);
        Diff.Check("ScatterND.fold", tg.B.Build(), new Step(1, 0, (x, Rnd.F(512, 3))), atol: 0, rtol: 0);
    }

    /// <summary>Ten scatters with updates in ten weight buffers (two gather passes) writing overlapping rows: the later scatter wins.</summary>
    [Fact]
    public void ManySourcesOverlapping()
    {
        var tg = new TG();
        var x = tg.InF("x", 6, 5);
        var y = x;
        for (int k = 0; k < 10; k++)
        {
            var update = tg.RandW($"u{k}", DataType.F32, 1f, 1, 5);
            y = tg.B.Emit1("ScatterND", [y, IndexConstant(tg, $"idx{k}", [k % 6], 1, 1), update]);
        }

        tg.B.Output("y", y);
        Diff.Check("ScatterND.passes", tg.B.Build(), new Step(1, 0, (x, Rnd.F(30, 4))), atol: 0, rtol: 0);
    }

    /// <summary>An intermediate read by another node ends the chain there; negative indices.</summary>
    [Fact]
    public void EscapingIntermediate()
    {
        var tg = new TG();
        var x = tg.InF("x", 4, 3);
        var u = tg.InF("u", 2, 3);
        var first = tg.B.Emit1("ScatterND", [x, IndexConstant(tg, "i0", [-1, 1], 2, 1), u]);
        var second = tg.B.Emit1("ScatterND", [first, IndexConstant(tg, "i1", [0, -3], 2, 1), u]);
        tg.B.Output("first", tg.B.Emit1("Add", [first, first]));
        tg.B.Output("second", second);
        Diff.Check("ScatterND.escape", tg.B.Build(), new Step(1, 0, (x, Rnd.F(12, 5)), (u, Rnd.F(6, 6))), atol: 0, rtol: 0);
    }

    /// <summary>Runtime indices addressing single elements (q = rank): one thread per update element.</summary>
    [Fact]
    public void RuntimeElementIndices()
    {
        var tg = new TG();
        var x = tg.InF("x", 3, 4, 5);
        var idx = tg.In("idx", DataType.I32, 6, 3);
        var u = tg.InF("u", 6);
        tg.B.Output("y", tg.B.Emit1("ScatterND", [x, idx, u]));
        int[] iv = [0, 0, 0, 2, 3, 4, 1, 2, 3, -1, 0, 1, 0, -2, 2, 2, 1, 0];
        Diff.Check("ScatterND.runtime", tg.B.Build(), new Step(1, 0, (x, Rnd.F(60, 7)), (idx, iv), (u, Rnd.F(6, 8))), atol: 0, rtol: 0);
    }

    /// <summary>Integer data with i64 constant indices (bit-exact raw copies).</summary>
    [Fact]
    public void IntegerDataInt64Indices()
    {
        var tg = new TG();
        var x = tg.In("x", DataType.I32, 4, 6);
        var u = tg.In("u", DataType.I32, 3, 6);
        long[] indices = [3, 0, 2];
        var ix = tg.B.Constant("idx", DataType.I64, new Shape([3, 1]), MemoryMarshal.AsBytes(indices.AsSpan()).ToArray());
        tg.B.Output("y", tg.B.Emit1("ScatterND", [x, ix, u]));
        Diff.Check("ScatterND.int", tg.B.Build(), new Step(1, 0, (x, Rnd.I(24, 9, 1000)), (u, Rnd.I(18, 10, 1000))), atol: 0, rtol: 0);
    }
}
