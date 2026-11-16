using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>The time-major ops of the speech models (ConvTranspose1d, Interpolate1d, Lstm, reflect-padded Conv1d, Atan2) against direct implementations.</summary>
public class SpeechOpsTests
{
    [Theory]
    [InlineData(7, 6, 4, 4, 2, 1, 0, 1, true)]
    [InlineData(5, 8, 8, 3, 2, 1, 1, 8, true)]
    [InlineData(9, 4, 6, 20, 10, 5, 0, 1, false)]
    [InlineData(6, 4, 4, 12, 6, 3, 0, 1, true)]
    public void ConvTranspose1d_MatchesDirectScatter(int time, int channels, int outputs, int taps, int stride, int cropLeft, int outputPadding, int groups, bool bias)
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, channels);
        var weights = Kit.Rand(channels * (outputs / groups) * taps, 3);
        var biases = bias ? Kit.Rand(outputs, 4) : new float[outputs];
        var inputs = new List<Value> { x, tg.WF("w", weights, channels, outputs / groups, taps) };
        if (bias)
        {
            inputs.Add(tg.WF("b", biases, outputs));
        }

        var attributes = Attributes.From(("stride", (long)stride), ("crop_left", (long)cropLeft), ("crop_right", (long)cropLeft), ("output_padding", (long)outputPadding), ("groups", (long)groups));
        tg.B.Output("y", tg.B.Emit1("ConvTranspose1d", inputs, attributes));
        using var m = Kit.Compile(tg.B.Build());
        var input = Kit.Rand(time * channels, 5);
        Kit.Set(m, x, input);
        var p = Kit.P(time);
        m.Run(p);
        var actual = Kit.Out(m, "y", p);

        int full = ((time - 1) * stride) + taps + outputPadding;
        int outLength = full - (2 * cropLeft);
        var expected = new float[outLength * outputs];
        int perGroup = channels / groups;
        int outPerGroup = outputs / groups;
        for (int u = 0; u < outLength; u++)
        {
            for (int o = 0; o < outputs; o++)
            {
                expected[(u * outputs) + o] = biases[o];
            }
        }

        for (int t = 0; t < time; t++)
        {
            for (int k = 0; k < taps; k++)
            {
                int u = (t * stride) + k - cropLeft;
                if (u < 0 || u >= outLength)
                {
                    continue;
                }

                for (int o = 0; o < outputs; o++)
                {
                    int g = o / outPerGroup;
                    for (int c = 0; c < perGroup; c++)
                    {
                        int ci = (g * perGroup) + c;
                        expected[(u * outputs) + o] += input[(t * channels) + ci] * weights[(((ci * outPerGroup) + (o % outPerGroup)) * taps) + k];
                    }
                }
            }
        }

        Kit.AssertClose(expected, actual, 1e-4, 1e-4);
    }

    [Theory]
    [InlineData("nearest", 2, 1, 5)]
    [InlineData("nearest", 300, 1, 3)]
    [InlineData("linear", 300, 1, 4)]
    [InlineData("linear", 1, 300, 1200)]
    [InlineData("linear", 3, 2, 8)]
    public void Interpolate1d_FollowsTheHalfPixelRule(string mode, int up, int down, int time)
    {
        const int channels = 3;
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, channels);
        tg.B.Output("y", tg.B.Emit1("Interpolate1d", [x], Attributes.From(("mode", mode), ("up", (long)up), ("down", (long)down))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 4096);
        var input = Kit.Rand(time * channels, 5);
        Kit.Set(m, x, input);
        var p = Kit.P(time);
        m.Run(p);
        var actual = Kit.Out(m, "y", p);

        int outLength = time * up / down;
        var expected = new float[outLength * channels];
        for (int j = 0; j < outLength; j++)
        {
            for (int c = 0; c < channels; c++)
            {
                if (mode == "nearest")
                {
                    expected[(j * channels) + c] = input[(Math.Min(time - 1, j * down / up) * channels) + c];
                    continue;
                }

                double source = Math.Max(((j + 0.5) * down / up) - 0.5, 0);
                int i0 = Math.Min((int)Math.Floor(source), time - 1);
                int i1 = Math.Min(i0 + 1, time - 1);
                double a = input[(i0 * channels) + c];
                double b = input[(i1 * channels) + c];
                expected[(j * channels) + c] = (float)(a + ((b - a) * (source - i0)));
            }
        }

        Kit.AssertClose(expected, actual, 1e-5, 1e-5);
    }

    [Theory]
    [InlineData(1, 8, false)]
    [InlineData(13, 16, false)]
    [InlineData(13, 16, true)]
    [InlineData(40, 300, true)]
    public void Lstm_MatchesDirectRecurrence(int time, int hidden, bool reverse)
    {
        var tg = new TestGraph();
        var gates = tg.InF("g", TestGraph.T, 4 * hidden);
        var weights = Kit.Rand(4 * hidden * hidden, 3).Select(v => v * 0.3f).ToArray();
        var w = tg.WF("w", weights, 4 * hidden, hidden);
        tg.B.Output("y", tg.B.Emit1("Lstm", [gates, w], Attributes.From(("reverse", reverse))));
        using var m = Kit.Compile(tg.B.Build());
        var input = Kit.Rand(time * 4 * hidden, 5);
        Kit.Set(m, gates, input);
        var p = Kit.P(time);
        m.Run(p);
        var actual = Kit.Out(m, "y", p);

        var expected = new float[time * hidden];
        var h = new double[hidden];
        var c = new double[hidden];
        static double Sig(double v) => 1 / (1 + Math.Exp(-v));
        for (int s = 0; s < time; s++)
        {
            int t = reverse ? time - 1 - s : s;
            var z = new double[4 * hidden];
            for (int r = 0; r < 4 * hidden; r++)
            {
                double sum = input[(t * 4 * hidden) + r];
                for (int k = 0; k < hidden; k++)
                {
                    sum += weights[(r * hidden) + k] * h[k];
                }

                z[r] = sum;
            }

            for (int j = 0; j < hidden; j++)
            {
                c[j] = (Sig(z[hidden + j]) * c[j]) + (Sig(z[j]) * Math.Tanh(z[(2 * hidden) + j]));
                h[j] = Sig(z[(3 * hidden) + j]) * Math.Tanh(c[j]);
                expected[(t * hidden) + j] = (float)h[j];
            }
        }

        Kit.AssertClose(expected, actual, 1e-4, 1e-4);
    }

    [Fact]
    public void Conv1d_ReflectPadding_MirrorsWithoutRepeatingTheEdge()
    {
        const int time = 9;
        const int taps = 5;
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 1);
        var weights = Kit.Rand(taps, 3);
        var attributes = Attributes.From(("pad_left", 2L), ("pad_right", 2L), ("pad_mode", "reflect"));
        tg.B.Output("y", tg.B.Emit1("Conv1d", [x, tg.WF("w", weights, 1, 1, taps)], attributes));
        using var m = Kit.Compile(tg.B.Build());
        var input = Kit.Rand(time, 5);
        Kit.Set(m, x, input);
        var p = Kit.P(time);
        m.Run(p);
        var actual = Kit.Out(m, "y", p);

        var expected = new float[time];
        for (int t = 0; t < time; t++)
        {
            double sum = 0;
            for (int k = 0; k < taps; k++)
            {
                int ti = t - 2 + k;
                ti = ti < 0 ? -ti : ti >= time ? (2 * (time - 1)) - ti : ti;
                sum += input[ti] * weights[k];
            }

            expected[t] = (float)sum;
        }

        Kit.AssertClose(expected, actual, 1e-5, 1e-5);
    }

    [Fact]
    public void Atan2_IsTheAngleOfTheSecondOperandAndTheFirst()
    {
        var tg = new TestGraph();
        var y = tg.InF("y", TestGraph.T, 2);
        var x = tg.InF("x", TestGraph.T, 2);
        tg.B.Output("a", tg.B.Binary("Atan2", y, x));
        using var m = Kit.Compile(tg.B.Build());
        float[] ys = [1f, -1f, 0.5f, -2f, 0f, 0f];
        float[] xs = [1f, 1f, -1f, -3f, -1f, 0.5f];
        Kit.Set(m, y, ys);
        Kit.Set(m, x, xs);
        var p = Kit.P(3);
        m.Run(p);
        var actual = Kit.Out(m, "a", p);
        Kit.AssertClose(ys.Select((v, i) => MathF.Atan2(v, xs[i])).ToArray(), actual, 1e-5, 1e-5);
    }
}
