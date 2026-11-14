using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// Det, Inverse, volumetric GridSample and int64 CumSum compute on Vulkan what they compute on the CPU, over a symbolic batch.
/// </summary>
public sealed class DeviceLinalgOpsTests
{
    private const int Batch = 3;

    private static TestModel DetInverse()
        => new TestModel().Opset(13).Opset(1, "com.microsoft").Input("x", "N", 4, 5, 5).Output("d", "N", 4).Output("i", "N", 4, 5, 5)
            .Node("Det", ["x"], ["d"]).ContribNode("Inverse", ["x"], ["i"]);

    private static TestModel GridSample3D(string mode)
        => new TestModel().Opset(20).Input("x", "N", 2, 4, 5, 6).Input("g", "N", 3, 4, 5, 3).Output("y", "N", 2, 3, 4, 5)
            .Node("GridSample", ["x", "g"], ["y"], TestModel.AttrS("mode", mode), TestModel.AttrS("padding_mode", "border"));

    /// <summary>x → int64 (truncating) → CumSum along axis 1 → float.</summary>
    private static TestModel CumSumInt64()
        => new TestModel().Opset(14).Input("x", "N", 9, 4).Output("y", "N", 9, 4).InitI64("axis", [], 1)
            .Node("Cast", ["x"], ["xi"], TestModel.Attr("to", 7)).Node("CumSum", ["xi", "axis"], ["ci"], TestModel.Attr("reverse", 1))
            .Node("Cast", ["ci"], ["y"], TestModel.Attr("to", 1));

    [Fact]
    public void DetAndInverse_MatchCpu()
    {
        // Diagonally dominant matrices: well conditioned, so f32 (Vulkan) and f64 (CPU) elimination agree closely.
        var x = Rnd(Batch * 4 * 25, 1);
        for (int m = 0; m < Batch * 4; m++)
        {
            for (int r = 0; r < 5; r++)
            {
                x[m * 25 + r * 5 + r] += 3f;
            }
        }

        Compare(DetInverse(), [x], 1e-4);
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("nearest")]
    public void VolumetricGridSample_MatchesCpu(string mode)
        => Compare(GridSample3D(mode), [Rnd(Batch * 2 * 4 * 5 * 6, 2), [.. Rnd(Batch * 3 * 4 * 5 * 3, 3).Select(v => v * 2.4f)]], 1e-5);

    [Fact]
    public void Int64CumSum_MatchesCpu()
        => Compare(CumSumInt64(), [[.. Rnd(Batch * 9 * 4, 4).Select(v => v * 200f)]], 0);

    private static void Compare(TestModel model, List<float[]> inputs, double tolerance)
    {
        var vulkan = Providers.Get("vulkan");
        string path = Path.Combine(Path.GetTempPath(), $"sl-device-linalg-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, model.ToBytes());
        try
        {
            var expected = OnnxExec.Run(path, new OnnxImportOptions(), Providers.Cpu, Batch, inputs).Outputs;
            var actual = OnnxExec.Run(path, new OnnxImportOptions(), vulkan, Batch, inputs).Outputs;
            Assert.Equal(expected.Length, actual.Length);
            for (int o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o].Length, actual[o].Length);
                for (int i = 0; i < expected[o].Length; i++)
                {
                    Assert.True(Math.Abs(actual[o][i] - expected[o][i]) <= tolerance * (1 + Math.Abs(expected[o][i])),
                        $"output {o} [{i}] vulkan {actual[o][i]} vs cpu {expected[o][i]}");
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
