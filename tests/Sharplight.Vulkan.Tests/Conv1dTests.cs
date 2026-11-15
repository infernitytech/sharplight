using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>The Conv1d op on the device against the CPU provider, over a symbolic time axis.</summary>
public sealed class Conv1dTests
{
    [Theory]
    [InlineData(12, 20, 7, 1, 1, 3, 3, 1, true, "f32")]
    [InlineData(10, 10, 3, 1, 1, 1, 1, 1, true, "f16")]
    [InlineData(8, 8, 7, 1, 1, 3, 3, 8, true, "f16")]
    [InlineData(6, 12, 4, 1, 1, 2, 1, 2, false, "f32")]
    [InlineData(5, 5, 3, 1, 3, 3, 3, 5, true, "f32")]
    public void MatchesTheCpu(int channels, int outputs, int taps, int stride, int dilation, int padLeft, int padRight, int groups, bool bias, string wtype)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, channels);
        var inputs = new List<Value> { x, tg.RandW("w", DataType.Get(wtype), 0.4f, outputs, channels / groups, taps) };
        if (bias)
        {
            inputs.Add(tg.RandW("b", DataType.F32, 1f, outputs));
        }

        var attributes = Attributes.From(("stride", (long)stride), ("pad_left", (long)padLeft), ("pad_right", (long)padRight), ("dilation", (long)dilation), ("groups", (long)groups));
        tg.B.Output("y", tg.B.Emit1("Conv1d", inputs, attributes));
        var steps = new[] { 1, 17, 64 }.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * channels, 30 + i)))).ToList();
        Diff.Check("Conv1d", tg.B.Build(), steps, 2e-4, 1e-4, maxTokens: 64, what: $"{channels}->{outputs} k{taps} g{groups} {wtype}");
    }
}
