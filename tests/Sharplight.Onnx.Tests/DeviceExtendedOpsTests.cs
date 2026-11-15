using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// LinearAttention, DeformConv, Unique and long DFTs compute on Vulkan, without the CPU fallback, what they compute on the CPU, over a
/// symbolic batch.
/// </summary>
public sealed class DeviceExtendedOpsTests
{
    private const int Batch = 3;

    /// <summary>Gated delta rule with grouped query heads (8 query heads over 4 key/value heads), per-key-dim decay and a past state.</summary>
    private static TestModel GatedDeltaGqa()
        => new TestModel().Opset(27).Input("q", "N", 5, 64).Input("k", "N", 5, 32).Input("v", "N", 5, 32).Input("s", "N", 4, 8, 8)
            .Input("g", "N", 5, 32).Input("b", "N", 5, 4).Output("y", "N", 5, 64).Output("sy", "N", 4, 8, 8)
            .Node("LinearAttention", ["q", "k", "v", "s", "g", "b"], ["y", "sy"], TestModel.Attr("q_num_heads", 8), TestModel.Attr("kv_num_heads", 4));

    private static TestModel Linear()
        => new TestModel().Opset(27).Input("q", "N", 6, 16).Input("k", "N", 6, 16).Input("v", "N", 6, 8).Output("y", "N", 6, 8).Output("sy", "N", 2, 8, 4)
            .Node("LinearAttention", ["q", "k", "v"], ["y", "sy"], TestModel.Attr("q_num_heads", 2), TestModel.Attr("kv_num_heads", 2),
                TestModel.AttrS("update_rule", "linear"));

    /// <summary>Two groups and two offset groups, stride 2 with padding and dilation, a mask and a bias.</summary>
    private static TestModel DeformConv()
        => new TestModel().Opset(22).Input("x", "N", 4, 7, 6).Input("off", "N", 36, 4, 3).Input("m", "N", 18, 4, 3).Output("y", "N", 6, 4, 3)
            .InitF32("w", [6, 2, 3, 3], Rnd(6 * 2 * 9, 7)).InitF32("bias", [6], Rnd(6, 8))
            .Node("DeformConv", ["x", "w", "off", "bias", "m"], ["y"], TestModel.AttrInts("kernel_shape", 3, 3), TestModel.AttrInts("strides", 2, 2),
                TestModel.AttrInts("pads", 1, 1, 1, 1), TestModel.AttrInts("dilations", 1, 2), TestModel.Attr("group", 2), TestModel.Attr("offset_group", 2));

    private static TestModel Unique(int sorted)
        => new TestModel().Opset(11).Input("x", "N", 7).Output("u", "U").Output("i", "U").Output("inv", "M").Output("c", "U")
            .Node("Unique", ["x"], ["u", "i", "inv", "c"], TestModel.Attr("sorted", sorted));

    /// <summary>A real DFT of 1500 points (too long for the importer's dense basis), onesided.</summary>
    private static TestModel LongDft()
        => new TestModel().Opset(20).Input("x", "N", 1500, 1).Output("y", "N", 751, 2).InitI64("len", [], 1500).InitI64("axis", [], 1)
            .Node("DFT", ["x", "len", "axis"], ["y"], TestModel.Attr("onesided", 1));

    [Fact]
    public void GatedDeltaRuleWithGroupedHeads_MatchesCpu()
    {
        var beta = Rnd(Batch * 5 * 4, 6).Select(v => v + 0.5f).ToArray();
        var decay = Rnd(Batch * 5 * 32, 5).Select(v => v - 0.3f).ToArray();
        Compare(GatedDeltaGqa(), [Rnd(Batch * 5 * 64, 1), Rnd(Batch * 5 * 32, 2), Rnd(Batch * 5 * 32, 3), Rnd(Batch * 4 * 64, 4), decay, beta], 1e-5);
    }

    [Fact]
    public void LinearRule_MatchesCpu() => Compare(Linear(), [Rnd(Batch * 6 * 16, 1), Rnd(Batch * 6 * 16, 2), Rnd(Batch * 6 * 8, 3)], 1e-5);

    [Fact]
    public void DeformConvWithGroupsAndMask_MatchesCpu()
        => Compare(DeformConv(), [Rnd(Batch * 4 * 7 * 6, 1), [.. Rnd(Batch * 36 * 12, 2).Select(v => v * 3f)], Rnd(Batch * 18 * 12, 3)], 1e-5);

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void Unique_MatchesCpu(int sorted)
    {
        // Repeated values (and a NaN pair) across the batch.
        float[] x = [.. Enumerable.Range(0, Batch * 7).Select(i => (float)((i * 5) % 6) - 2f)];
        x[4] = float.NaN;
        x[11] = float.NaN;
        Compare(Unique(sorted), [x], 0);
    }

    [Fact]
    public void LongRealDft_MatchesCpu() => Compare(LongDft(), [Rnd(Batch * 1500, 1)], 2e-4);

    private static void Compare(TestModel model, List<float[]> inputs, double tolerance)
    {
        var vulkan = Providers.Get("vulkan");
        string path = Path.Combine(Path.GetTempPath(), $"sl-device-extended-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, model.ToBytes());
        try
        {
            var expected = OnnxExec.Run(path, new OnnxImportOptions(), Providers.Cpu, Batch, inputs).Outputs;
            var actual = OnnxExec.Run(path, new OnnxImportOptions(), vulkan, Batch, inputs, fallback: false).Outputs;
            Assert.Equal(expected.Length, actual.Length);
            for (int o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o].Length, actual[o].Length);
                double scale = 1 + expected[o].Where(float.IsFinite).Select(v => Math.Abs((double)v)).DefaultIfEmpty(0).Max();
                for (int i = 0; i < expected[o].Length; i++)
                {
                    bool same = float.IsNaN(expected[o][i]) ? float.IsNaN(actual[o][i]) : Math.Abs(actual[o][i] - expected[o][i]) <= tolerance * scale;
                    Assert.True(same, $"output {o} [{i}] vulkan {actual[o][i]} vs cpu {expected[o][i]}");
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
