using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>The Conv1d op over a symbolic time axis against a direct convolution.</summary>
public class Conv1dTests
{
    [Theory]
    [InlineData(1, 12, 20, 7, 1, 1, 3, 3, 1, true)]
    [InlineData(9, 6, 10, 3, 1, 1, 1, 1, 1, true)]
    [InlineData(17, 8, 8, 7, 1, 1, 3, 3, 8, true)]
    [InlineData(23, 6, 12, 4, 2, 1, 2, 1, 2, false)]
    [InlineData(30, 5, 5, 3, 1, 3, 3, 3, 5, true)]
    public void Conv1d_MatchesDirectConvolution(int time, int channels, int outputs, int taps, int stride, int dilation, int padLeft, int padRight, int groups, bool bias)
    {
        var tg = new TestGraph();
        var biases = bias ? Kit.Rand(outputs, 4) : new float[outputs];
        var x = tg.InF("x", TestGraph.T, channels);
        var weights = Kit.Rand(outputs * (channels / groups) * taps, 3);
        var w = tg.WF("w", weights, outputs, channels / groups, taps);
        var inputs = new List<Value> { x, w };
        if (bias)
        {
            inputs.Add(tg.WF("b", biases, outputs));
        }

        var attributes = Attributes.From(("stride", (long)stride), ("pad_left", (long)padLeft), ("pad_right", (long)padRight), ("dilation", (long)dilation), ("groups", (long)groups));
        var y = tg.B.Emit1("Conv1d", inputs, attributes);
        tg.B.Output("y", y);
        using var m = Kit.Compile(tg.B.Build());
        var input = Kit.Rand(time * channels, 5);
        Kit.Set(m, x, input);
        var p = Kit.P(time);
        m.Run(p);
        var actual = Kit.Out(m, "y", p);

        int outLength = ((time + padLeft + padRight - (dilation * (taps - 1)) - 1) / stride) + 1;
        var expected = new float[outLength * outputs];
        int perGroup = channels / groups;
        int outPerGroup = outputs / groups;
        for (int t = 0; t < outLength; t++)
        {
            for (int o = 0; o < outputs; o++)
            {
                double sum = biases[o];
                for (int k = 0; k < taps; k++)
                {
                    int ti = (t * stride) - padLeft + (k * dilation);
                    if (ti < 0 || ti >= time)
                    {
                        continue;
                    }

                    for (int c = 0; c < perGroup; c++)
                    {
                        sum += input[(ti * channels) + (o / outPerGroup * perGroup) + c] * weights[(((o * perGroup) + c) * taps) + k];
                    }
                }

                expected[(t * outputs) + o] = (float)sum;
            }
        }

        Kit.AssertClose(expected, actual, 1e-4, 1e-4);
    }
}
