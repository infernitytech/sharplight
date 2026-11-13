using System.Diagnostics;
using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>Synthetic ResNet-50 (random weights, BN folded into conv bias) throughput; run with SHARPLIGHT_BENCH=1.</summary>
public class SpatialBenchTests
{
    private static void Report(string s)
    {
        Console.Error.WriteLine(s);
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_OUT") is { Length: > 0 } f)
        {
            File.AppendAllText(f, s + Environment.NewLine);
        }
    }

    private sealed class Net
    {
        public readonly TG Tg = new();
        public double Flops; private int _id; private readonly Random _rng = new(7);
        public Value Conv(Value x, int cin, int cout, int k, int stride, int pad, int hw, bool relu, out int ohw, int groups = 1)
        {
            int n = cout * (cin / groups) * k * k;
            var w = new float[n]; float sc = MathF.Sqrt(2f / (cin / groups * k * k));
            for (int i = 0; i < n; i++)
            {
                w[i] = (float)(_rng.NextDouble() * 2 - 1) * sc;
            }

            var b = new float[cout]; for (int i = 0; i < cout; i++)
            {
                b[i] = (float)(_rng.NextDouble() - 0.5) * 0.1f;
            }

            int id = _id++;
            var wv = Tg.WF($"w{id}", w, cout, cin / groups, k, k); var bv = Tg.WF($"b{id}", b, cout);
            ohw = (hw + 2 * pad - k) / stride + 1;
            Flops += 2.0 * cout * (cin / groups) * k * k * ohw * ohw;
            var y = Tg.B.Emit1("Conv", [x, wv, bv], Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
            return relu ? Tg.B.Unary("Relu", y) : y;
        }

        public Value Bottleneck(Value x, int cin, int mid, int cout, int stride, int hw, out int ohw)
        {
            var a = Conv(x, cin, mid, 1, 1, 0, hw, true, out _);
            var b = Conv(a, mid, mid, 3, stride, 1, hw, true, out ohw);
            var c = Conv(b, mid, cout, 1, 1, 0, ohw, false, out _);
            var sc = (stride != 1 || cin != cout) ? Conv(x, cin, cout, 1, stride, 0, hw, false, out _) : x;
            return Tg.B.Unary("Relu", Tg.B.Add(c, sc));
        }
    }

    [Theory]
    [InlineData(1), InlineData(8)]
    public void ResNet50(int batch)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            Par.Threads = thr;
        }

        var net = new Net(); var tg = net.Tg;
        var x = tg.InF("x", TG.T, 3, 224, 224);
        var y = net.Conv(x, 3, 64, 7, 2, 3, 224, true, out int hw);
        y = tg.B.Emit1("Pool", [y], Attributes.From(("mode", "max"), ("kernel", new long[] { 3, 3 }), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 })));
        hw = (hw + 2 - 3) / 2 + 1;
        int cin = 64;
        foreach (var (mid, blocks, stride) in new[] { (64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2) })
        {
            for (int b = 0; b < blocks; b++)
            {
                y = net.Bottleneck(y, cin, mid, mid * 4, b == 0 ? stride : 1, hw, out hw);
                cin = mid * 4;
            }
        }

        y = tg.B.Emit1("Reduce", [y], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", false)));
        var fw = tg.WF("fc", Kit.Rand(1000 * 2048, 3, 0.02f), 1000, 2048);
        tg.B.Output("logits", tg.B.MatMul(y, fw));
        net.Flops += 2.0 * 1000 * 2048;
        using var m = Kit.Compile(tg.B.Build(), maxTokens: batch);
        Kit.Set(m, x, Kit.Rand(batch * 3 * 224 * 224, 5));
        var p = Kit.P(batch);
        for (int i = 0; i < 5; i++)
        {
            m.Run(p);
        }

        int iters = 30; var times = new double[iters];
        for (int i = 0; i < iters; i++) { long t0 = Stopwatch.GetTimestamp(); m.Run(p); times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds; }
        Array.Sort(times);
        double med = times[iters / 2], best = times[0];
        var logits = Kit.Out(m, "logits", p);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Report($"ResNet-50 batch {batch}: best {best * 1e3:F2} ms, median {med * 1e3:F2} ms  ({net.Flops * batch / best / 1e9:F0} GFLOP/s best, {net.Flops / 1e9:F2} GFLOP/img, threads {Par.Threads})");
    }
}
