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
        public readonly TestGraph Tg = new();
        public double Flops;
        private int _id;
        private readonly Random _rng = new(7);
        public Value Conv(Value x, int cin, int cout, int k, int stride, int pad, int hw, bool relu, out int ohw, int groups = 1)
        {
            int n = cout * (cin / groups) * k * k;
            var w = new float[n];
            float sc = MathF.Sqrt(2f / (cin / groups * k * k));
            for (int i = 0; i < n; i++)
            {
                w[i] = (float)(_rng.NextDouble() * 2 - 1) * sc;
            }

            var b = new float[cout];
            for (int i = 0; i < cout; i++)
            {
                b[i] = (float)(_rng.NextDouble() - 0.5) * 0.1f;
            }

            int id = _id++;
            var wv = Tg.WF($"w{id}", w, cout, cin / groups, k, k);
            var bv = Tg.WF($"b{id}", b, cout);
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
            CpuParallel.Threads = thr;
        }

        var net = new Net();
        var tg = net.Tg;
        var x = tg.InF("x", TestGraph.T, 3, 224, 224);
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

        int iters = 30;
        var times = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            m.Run(p);
            times[i] = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        }
        Array.Sort(times);
        double med = times[iters / 2];
        double best = times[0];
        var logits = Kit.Out(m, "logits", p);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Report($"ResNet-50 batch {batch}: best {best * 1e3:F2} ms, median {med * 1e3:F2} ms  ({net.Flops * batch / best / 1e9:F0} GFLOP/s best, {net.Flops / 1e9:F2} GFLOP/img, threads {CpuParallel.Threads})");
    }

    /// <summary>
    /// GFLOP/s of single convolutions with the distinct ResNet-50 shapes (plus a few YOLO / EfficientNet ones) at batch 1 and
    /// 8; run with SHARPLIGHT_BENCH=1. Each line: Cin, Cout, kernel, stride, input size, best time, GFLOP/s.
    /// </summary>
    [Fact]
    public void ConvShapes()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
        if (int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_THREADS"), out var thr))
        {
            CpuParallel.Threads = thr;
        }

        (int Cin, int Cout, int K, int Stride, int Hw)[] shapes =
        [
            (3, 64, 7, 2, 224), (64, 64, 1, 1, 56), (64, 64, 3, 1, 56), (64, 256, 1, 1, 56), (256, 64, 1, 1, 56),
            (256, 128, 1, 1, 56), (128, 128, 3, 2, 56), (128, 128, 3, 1, 28), (128, 512, 1, 1, 28), (512, 128, 1, 1, 28),
            (256, 512, 1, 2, 56), (512, 256, 1, 1, 28), (256, 256, 3, 1, 14), (256, 1024, 1, 1, 14), (1024, 256, 1, 1, 14),
            (1024, 512, 1, 1, 14), (512, 512, 3, 1, 7), (512, 2048, 1, 1, 7), (2048, 512, 1, 1, 7), (1024, 2048, 1, 2, 14),
            (16, 32, 3, 2, 320), (32, 32, 3, 1, 160), (64, 64, 3, 1, 80), (128, 128, 3, 1, 40),
        ];
        string? only = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_SHAPE");
        foreach (int batch in new[] { 1, 8 })
        {
            double totalTime = 0;
            double totalFlops = 0;
            foreach (var (cin, cout, k, stride, hw) in shapes)
            {
                string key = $"{cin}x{cout}k{k}s{stride}@{hw}";
                if (only is { Length: > 0 } && !only.Split(',').Contains(key))
                {
                    continue;
                }

                var net = new Net();
                var tg = net.Tg;
                var x = tg.InF("x", TestGraph.T, cin, hw, hw);
                var y = net.Conv(x, cin, cout, k, stride, k / 2, hw, true, out int ohw);
                tg.B.Output("y", y);
                using var m = Kit.Compile(tg.B.Build(), maxTokens: batch);
                Kit.Set(m, x, Kit.Rand(batch * cin * hw * hw, 5));
                var p = Kit.P(batch);
                for (int i = 0; i < 3; i++)
                {
                    m.Run(p);
                }

                double best = double.MaxValue;
                long budget = Stopwatch.GetTimestamp() + Stopwatch.Frequency / 2;
                for (int i = 0; i < 200 && (i < 10 || Stopwatch.GetTimestamp() < budget); i++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    m.Run(p);
                    best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds);
                }

                double flops = net.Flops * batch;
                totalTime += best;
                totalFlops += flops;
                Report($"b{batch} {key,-18} {best * 1e3,8:F3} ms {flops / best / 1e9,7:F0} GFLOP/s");
            }

            Report($"b{batch} total {totalTime * 1e3:F2} ms, {totalFlops / totalTime / 1e9:F0} GFLOP/s");
        }
    }
}
