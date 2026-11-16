using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>The time-major speech ops on the device against the CPU provider, over a symbolic time axis.</summary>
public sealed class SpeechOpsTests
{
    [Theory]
    [InlineData(6, 6, 4, 2, 1, 0, 1, true)]
    [InlineData(8, 8, 3, 2, 1, 1, 8, true)]
    [InlineData(4, 6, 20, 10, 5, 0, 1, false)]
    [InlineData(4, 4, 12, 6, 3, 0, 1, true)]
    public void ConvTranspose1d_MatchesTheCpu(int channels, int outputs, int taps, int stride, int cropLeft, int outputPadding, int groups, bool bias)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, channels);
        var inputs = new List<Value> { x, tg.RandW("w", DataType.F32, 0.4f, channels, outputs / groups, taps) };
        if (bias)
        {
            inputs.Add(tg.RandW("b", DataType.F32, 1f, outputs));
        }

        var attributes = Attributes.From(("stride", (long)stride), ("crop_left", (long)cropLeft), ("crop_right", (long)cropLeft), ("output_padding", (long)outputPadding), ("groups", (long)groups));
        tg.B.Output("y", tg.B.Emit1("ConvTranspose1d", inputs, attributes));
        var steps = new[] { 2, 9, 33 }.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * channels, 30 + i)))).ToList();
        Diff.Check("ConvTranspose1d", tg.B.Build(), steps, 2e-4, 1e-4, maxTokens: 64, what: $"{channels}->{outputs} k{taps} s{stride}");
    }

    [Theory]
    [InlineData("nearest", 2, 1)]
    [InlineData("linear", 300, 1)]
    [InlineData("linear", 1, 300)]
    [InlineData("linear", 3, 2)]
    public void Interpolate1d_MatchesTheCpu(string mode, int up, int down)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 3);
        tg.B.Output("y", tg.B.Emit1("Interpolate1d", [x], Attributes.From(("mode", mode), ("up", (long)up), ("down", (long)down))));
        var steps = new[] { 4, 7, 600 }.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * 3, 30 + i)))).ToList();
        Diff.Check("Interpolate1d", tg.B.Build(), steps, 1e-5, 1e-5, maxTokens: 600, what: $"{mode} {up}/{down}");
    }

    [Theory]
    [InlineData(8, false)]
    [InlineData(64, true)]
    [InlineData(300, false)]
    public void Lstm_MatchesTheCpu(int hidden, bool reverse)
    {
        Gpu.Require();
        var tg = new TG();
        var gates = tg.InF("g", TG.T, 4 * hidden);
        var w = tg.RandW("w", DataType.F32, 0.3f, 4 * hidden, hidden);
        tg.B.Output("y", tg.B.Emit1("Lstm", [gates, w], Attributes.From(("reverse", reverse))));
        var steps = new[] { 1, 11, 40 }.Select((t, i) => new Step(t, 0, (gates, Rnd.F((long)t * 4 * hidden, 30 + i)))).ToList();
        Diff.Check("Lstm", tg.B.Build(), steps, 5e-4, 1e-4, maxTokens: 64, what: $"h{hidden} reverse={reverse}");
    }

    [Fact]
    public void Conv1d_ReflectPadding_MatchesTheCpu()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4);
        var attributes = Attributes.From(("pad_left", 10L), ("pad_right", 10L), ("stride", 5L), ("pad_mode", "reflect"));
        tg.B.Output("y", tg.B.Emit1("Conv1d", [x, tg.RandW("w", DataType.F32, 0.4f, 6, 4, 20)], attributes));
        var steps = new[] { 30, 100 }.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * 4, 30 + i)))).ToList();
        Diff.Check("Conv1d", tg.B.Build(), steps, 2e-4, 1e-4, maxTokens: 100, what: "reflect");
    }

    [Fact]
    public void Atan2_MatchesTheCpu()
    {
        Gpu.Require();
        var tg = new TG();
        var y = tg.InF("y", TG.T, 5);
        var x = tg.InF("x", TG.T, 5);
        tg.B.Output("a", tg.B.Binary("Atan2", y, x));
        var steps = new[] { 3, 20 }.Select((t, i) => new Step(t, 0, (y, Rnd.F((long)t * 5, 30 + i)), (x, Rnd.F((long)t * 5, 50 + i)))).ToList();
        Diff.Check("Atan2", tg.B.Build(), steps, 1e-4, 1e-4, maxTokens: 20, what: "atan2");
    }
}
