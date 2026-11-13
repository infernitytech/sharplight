using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class PadTests
{
    private static void Run(string label, string mode, int[] shape, long[] pads, float value = 0f, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", [TG.T, .. shape.Select(d => (Dim)d)]);
        var xv = x;
        if (slice)
        {
            xv = tg.B.Slice(x, 2, 1, shape[1] - 2);
        }

        tg.B.Output("y", tg.B.Emit1("Pad", [xv], Attributes.From(("pads", pads), ("mode", mode), ("value", value))));
        long n = 3; foreach (var d in shape)
        {
            n *= d;
        }

        Diff.Check("Pad." + mode, tg.B.Build(), new Step(3, 0, (x, Rnd.F(n, 2))), what: label);
    }

    // pads layout: begin per axis (incl. the batch axis), then end per axis
    [Theory]
    [InlineData("constant")] [InlineData("reflect")] [InlineData("edge")]
    public void Spatial4D(string mode) => Run(mode, mode, [4, 9, 11], [0, 0, 2, 1, 0, 0, 3, 2], 1.5f);

    [Theory]
    [InlineData("constant")] [InlineData("reflect")] [InlineData("edge")]
    public void Rank3(string mode) => Run(mode, mode, [10, 7], [0, 1, 2, 0, 3, 1], -0.5f);

    [Fact] public void ConstantOnlyOneSide() => Run("one side", "constant", [3, 5, 6], [0, 0, 0, 2, 0, 0, 0, 0], 7f);
    [Fact] public void ConstantNegativeCrop() => Run("crop", "constant", [3, 8, 9], [0, 0, -1, 2, 0, 0, -2, 1]);
    [Fact] public void ReflectLargePad() => Run("reflect large", "reflect", [3, 4, 5], [0, 0, 0, 0, 0, 0, 5, 7]);
    [Fact] public void EdgeOnStridedInput() => Run("edge strided", "edge", [4, 9, 8], [0, 0, 1, 2, 0, 0, 2, 3], slice: true);
    [Fact] public void ReflectOnStridedInput() => Run("reflect strided", "reflect", [4, 9, 8], [0, 0, 1, 2, 0, 0, 2, 3], slice: true);
}
