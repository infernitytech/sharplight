using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// Concatenations along an inner axis whose inputs come from fused elementwise kernels: the GPU lowering lets those kernels write
/// straight into their (strided, unaligned) slices of the output. The result must match the CPU backend's, also when an input is a
/// graph input, a view, read twice or read by another node as well.
/// </summary>
public sealed class StridedConcatSliceTests
{
    private const int A = 3;
    private const int B = 5;
    private const int C = 6;

    /// <summary>
    /// x [N, A, B, C] → the rotary-style halves (x1 · c − x2 · s, x2 · c + x1 · s) of the last axis concatenated along
    /// <paramref name="axis"/>, with an extra input <paramref name="extra"/>: "input" (the graph input y), "view" (a Transpose of y),
    /// "shared" (an elementwise value that a second output reads too) or "twice" (one value concatenated twice).
    /// </summary>
    private static TestModel Model(int axis, string extra)
    {
        int half = C / 2;
        var m = new TestModel().Opset(13).Input("x", "N", A, B, C).Input("y", "N", A, B, C)
            .InitF32("c", [half], [0.5f, -0.25f, 0.75f]).InitF32("s", [half], [0.125f, 0.5f, -1f])
            .InitI64("split", [2], half, half)
            .Node("Split", ["x", "split"], ["x1", "x2"], TestModel.Attr("axis", 3))
            .Node("Mul", ["x1", "c"], ["x1c"]).Node("Mul", ["x2", "s"], ["x2s"]).Node("Sub", ["x1c", "x2s"], ["r1"])
            .Node("Mul", ["x2", "c"], ["x2c"]).Node("Mul", ["x1", "s"], ["x1s"]).Node("Add", ["x2c", "x1s"], ["r2"])
            .Node("Concat", ["r1", "r2"], ["r"], TestModel.Attr("axis", 3))
            .Node("Exp", ["r"], ["e"]);
        string[] parts = extra switch
        {
            "input" => ["e", "y"],
            "view" => ["e", "yt"],
            "shared" => ["e", "y2", "e2"],
            "twice" => ["e", "e2", "e"],
            _ => throw new ArgumentOutOfRangeException(nameof(extra)),
        };
        m.Node("Tanh", ["x"], ["e2"]).Node("Mul", ["y", "y"], ["y2"]);
        if (extra == "view")
        {
            m.Node("Transpose", ["y"], ["yt0"], TestModel.AttrInts("perm", 0, 2, 1, 3)).Node("Transpose", ["yt0"], ["yt"], TestModel.AttrInts("perm", 0, 2, 1, 3));
        }

        long[] dims = [A, B, C];
        dims[axis - 1] *= parts.Length;
        m.Node("Concat", parts, ["z"], TestModel.Attr("axis", axis)).Output("z", "N", dims[0], dims[1], dims[2]);
        if (extra == "shared")
        {
            m.Node("Neg", ["e2"], ["w"]).Output("w", "N", A, B, C);
        }

        return m;
    }

    [Theory]
    [InlineData(1, "input")]
    [InlineData(2, "input")]
    [InlineData(3, "input")]
    [InlineData(2, "view")]
    [InlineData(3, "shared")]
    [InlineData(2, "shared")]
    [InlineData(3, "twice")]
    public void GpuMatchesCpu(int axis, string extra)
    {
        var gpu = Providers.GpuOrSkip();
        const int batch = 2;
        var r = new Random(axis * 31 + extra.Length);
        float[] Rnd() => [.. Enumerable.Range(0, batch * A * B * C).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        List<float[]> inputs = [Rnd(), Rnd()];
        string path = Path.Combine(Path.GetTempPath(), $"sl-strided-concat-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, Model(axis, extra).ToBytes());
        try
        {
            var expected = OnnxExec.Run(path, new OnnxImportOptions(), Providers.Cpu, batch, inputs);
            var actual = OnnxExec.Run(path, new OnnxImportOptions(), gpu, batch, inputs);
            Assert.Equal(expected.Outputs.Length, actual.Outputs.Length);
            for (int o = 0; o < actual.Outputs.Length; o++)
            {
                var a = actual.Outputs[o];
                var b = expected.Outputs[o];
                Assert.Equal(b.Length, a.Length);
                for (int i = 0; i < a.Length; i++)
                {
                    Assert.True(Math.Abs(a[i] - b[i]) <= 1e-5 + 1e-5 * Math.Abs(b[i]), $"output {o} [{i}] gpu {a[i]} vs cpu {b[i]}");
                }
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
