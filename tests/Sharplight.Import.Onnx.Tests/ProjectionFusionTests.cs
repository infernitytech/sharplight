using Sharplight.Models;
using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// Several MatMuls of one activation by constant weights (with their bias Adds) become one MatMul over the concatenated weights,
/// and compute what the separate products compute, on the CPU and (when present) the GPU under test.
/// </summary>
public sealed class ProjectionFusionTests
{
    private const int S = 7;
    private const int K = 24;

    private static IEnumerable<string> Ops(ModelDefinition d) => d.Graph("main").Nodes.Select(n => n.Op.Name);

    /// <summary>Three projections of widths 16, 8 and 32; the second one's bias is the Add's first operand.</summary>
    private static TestModel Model(bool biases)
    {
        var m = new TestModel().Opset(13).Input("x", "N", S, K).Output("y0", "N", S, 16).Output("y1", "N", S, 8).Output("y2", "N", S, 32)
            .InitF32("w0", [K, 16], Rnd(K * 16, 1)).InitF32("w1", [K, 8], Rnd(K * 8, 2)).InitF32("w2", [K, 32], Rnd(K * 32, 3))
            .InitF32("b0", [16], Rnd(16, 4)).InitF32("b1", [8], Rnd(8, 5)).InitF32("b2", [32], Rnd(32, 6))
            .Node("MatMul", ["x", "w0"], ["p0"]).Node("MatMul", ["x", "w1"], ["p1"]).Node("MatMul", ["x", "w2"], ["p2"]);
        if (biases)
        {
            m.Node("Add", ["p0", "b0"], ["a0"]).Node("Add", ["b1", "p1"], ["a1"]).Node("Add", ["p2", "b2"], ["a2"]);
        }
        else
        {
            m.Node("Identity", ["p0"], ["a0"]).Node("Add", ["p1", "b1"], ["a1"]).Node("Identity", ["p2"], ["a2"]);
        }

        return m.Node("Relu", ["a0"], ["y0"]).Node("Sigmoid", ["a1"], ["y1"]).Node("Tanh", ["a2"], ["y2"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Siblings_AreOneMatMul(bool biases)
    {
        var ops = Ops(Model(biases).Build()).ToList();
        Assert.Single(ops, o => o == "MatMul");
        Assert.Equal(3, ops.Count(o => o == "Slice"));
        Assert.Equal(3, Ops(Model(biases).Build(new OnnxImportOptions { FuseProjections = false })).Count(o => o == "MatMul"));
    }

    [Theory]
    [InlineData("cpu", false)]
    [InlineData("cpu", true)]
    [InlineData("gpu", false)]
    [InlineData("gpu", true)]
    public void FusedMatchesSeparate(string provider, bool biases)
    {
        var device = Providers.Get(provider);
        const int batch = 3;
        List<float[]> inputs = [Rnd(batch * S * K, 11)];
        string path = Path.Combine(Path.GetTempPath(), $"sl-projections-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, Model(biases).ToBytes());
        try
        {
            var fused = OnnxExec.Run(path, new OnnxImportOptions(), device, batch, inputs);
            var plain = OnnxExec.Run(path, new OnnxImportOptions { FuseProjections = false }, device, batch, inputs);
            Assert.Equal(plain.Outputs.Length, fused.Outputs.Length);
            for (int o = 0; o < fused.Outputs.Length; o++)
            {
                var a = fused.Outputs[o];
                var b = plain.Outputs[o];
                Assert.Equal(b.Length, a.Length);
                for (int i = 0; i < a.Length; i++)
                {
                    Assert.True(Math.Abs(a[i] - b[i]) <= 1e-5 + 1e-5 * Math.Abs(b[i]), $"output {o} [{i}] fused {a[i]} vs separate {b[i]}");
                }
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static float[] Rnd(int n, int seed)
    {
        var r = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1) * 0.5f)];
    }
}
