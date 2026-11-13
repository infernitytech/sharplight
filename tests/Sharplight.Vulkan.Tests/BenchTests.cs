using System.Diagnostics;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;
using Xunit.Sdk;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// GPU micro-benchmarks (per-dispatch timestamps). Run with SHARPLIGHT_BENCH=1:
/// <c>SHARPLIGHT_BENCH=1 dotnet run --project tests/Sharplight.Vulkan.Tests -- -class Sharplight.Vulkan.Tests.BenchTests</c>
/// </summary>
public class BenchTests
{
    static void RequireBench()
    {
        Gpu.Require();
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1", "set SHARPLIGHT_BENCH=1 to run benchmarks");
    }

    static CompiledModel Compile(ComputeGraph g, long maxTokens = 8) => CompiledModel.Compile(g, new PlacementOptions
    {
        Devices = [new DeviceSlot { Device = Gpu.Vulkan! }],
        Compile = new CompileOptions { MaxTokens = maxTokens },
        ReservePerDevice = 0,
    });

    /// <summary>Runs the model repeatedly with profiling on; returns average GPU ms per execution and the report.</summary>
    static (double Ms, string Report) Profile(CompiledModel m, int n, int iters = 20)
    {
        var seg = (VulkanSegment)m.Segments[0];
        var p = new RunParameters(); p.Symbols["T"] = n; p.Scalars["n_past"] = 0;
        m.Run(p);   // warmup (pipelines)
        seg.Profile = true; seg.ResetProfile();
        for (int i = 0; i < iters; i++) m.Run(p);
        seg.Profile = false;
        double total = seg.ProfileData.Values.Sum(v => v.Ms) / iters;
        return (total, seg.ProfileReport(12));
    }

    static void Feed(CompiledModel m, Value x, long n) => m.SetInput(x, MemoryMarshal.AsBytes(Rnd.F(n, 1).AsSpan()));

    static void ConvBench(string label, int N, int C, int H, int W, int O, int k, int stride, int pad, int groups)
    {
        var tg = new TG(); var x = tg.InF("x", TG.T, C, H, W);
        var w = tg.RandW("w", DataType.F32, 0.1f, O, C / groups, k, k);
        var b = tg.RandW("b", DataType.F32, 0.1f, O);
        var y = tg.B.Emit1("Conv", [x, w, b], Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
        tg.B.Output("y", y);
        using var m = Compile(tg.B.Build());
        Feed(m, x, (long)N * C * H * W);
        var (ms, rep) = Profile(m, N);
        long HO = y.Shape[2].Fixed, WO = y.Shape[3].Fixed;
        double gflop = 2.0 * N * O * HO * WO * (C / groups) * k * k * 1e-9;
        Console.WriteLine($"BENCH {label,-44} {ms,8:F3} ms  {gflop / ms,8:F1} TFLOP/s  ({gflop:F3} GFLOP)");
    }

    [Fact]
    public void ConvLayers()
    {
        RequireBench();
        ConvBench("3x3 256->256 14x14", 1, 256, 14, 14, 256, 3, 1, 1, 1);
        ConvBench("3x3 256->256 14x14 N=8", 8, 256, 14, 14, 256, 3, 1, 1, 1);
        ConvBench("1x1 1024->256 14x14", 1, 1024, 14, 14, 256, 1, 1, 0, 1);
        ConvBench("1x1 256->1024 14x14", 1, 256, 14, 14, 1024, 1, 1, 0, 1);
        ConvBench("1x1 64->256 56x56", 1, 64, 56, 56, 256, 1, 1, 0, 1);
        ConvBench("1x1 256->64 56x56", 1, 256, 56, 56, 64, 1, 1, 0, 1);
        ConvBench("3x3 64->64 56x56", 1, 64, 56, 56, 64, 3, 1, 1, 1);
        ConvBench("3x3 128->128 28x28", 1, 128, 28, 28, 128, 3, 1, 1, 1);
        ConvBench("3x3 512->512 7x7", 1, 512, 7, 7, 512, 3, 1, 1, 1);
        ConvBench("1x1 2048->512 7x7", 1, 2048, 7, 7, 512, 1, 1, 0, 1);
        ConvBench("7x7 s2 3->64 224x224 (stem)", 1, 3, 224, 224, 64, 7, 2, 3, 1);
        ConvBench("dw 3x3 32ch 112x112", 1, 32, 112, 112, 32, 3, 1, 1, 32);
        ConvBench("dw 3x3 s2 144ch 56x56", 1, 144, 56, 56, 144, 3, 2, 1, 144);
        ConvBench("dw 3x3 960ch 14x14", 1, 960, 14, 14, 960, 3, 1, 1, 960);
        ConvBench("grouped 3x3 g32 128->128 28x28", 1, 128, 28, 28, 128, 3, 1, 1, 32);
        ConvBench("3x3 64->64 112x112 N=4", 4, 64, 112, 112, 64, 3, 1, 1, 1);
    }

    // ---- ResNet-50 (inference, BN folded into conv bias) ------------------------------------------
    sealed class Net(TG tg)
    {
        public TG G => tg;
        int _i;
        public Value Conv(Value x, int c, int o, int k, int s, int p, bool relu)
        {
            var w = tg.RandW($"w{_i}", DataType.F32, MathF.Sqrt(2f / (c * k * k)) * 1.7f, o, c, k, k);
            var b = tg.RandW($"b{_i++}", DataType.F32, 0.05f, o);
            var y = tg.B.Emit1("Conv", [x, w, b], Attributes.From(("strides", new long[] { s, s }), ("pads", new long[] { p, p, p, p })));
            return relu ? tg.B.Unary("Relu", y) : y;
        }
    }

    internal static ComputeGraph BuildResNet50(out TG tgOut, out Value input)
    {
        var tg = new TG("resnet50"); tgOut = tg;
        var n = new Net(tg);
        var x = tg.InF("image", TG.T, 3, 224, 224); input = x;
        var h = n.Conv(x, 3, 64, 7, 2, 3, true);
        h = tg.B.Emit1("Pool", [h], Attributes.From(("mode", "max"), ("kernel", new long[] { 3, 3 }), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 })));
        int c = 64;
        int[] blocks = [3, 4, 6, 3], widths = [64, 128, 256, 512];
        for (int stage = 0; stage < 4; stage++)
            for (int b = 0; b < blocks[stage]; b++)
            {
                int s = b == 0 && stage > 0 ? 2 : 1, w = widths[stage], o = w * 4;
                var y = n.Conv(h, c, w, 1, 1, 0, true);
                y = n.Conv(y, w, w, 3, s, 1, true);
                y = n.Conv(y, w, o, 1, 1, 0, false);
                var sc = b == 0 ? n.Conv(h, c, o, 1, s, 0, false) : h;
                h = tg.B.Unary("Relu", tg.B.Add(y, sc));
                c = o;
            }
        h = tg.B.Emit1("Reduce", [h], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", false)));
        var fc = tg.RandW("fc.w", DataType.F32, 0.05f, 1000, 2048);
        tg.B.Output("logits", tg.B.MatMul(h, fc));
        return tg.B.Build();
    }

    [Fact]
    public void ResNet50()
    {
        RequireBench();
        var g = BuildResNet50(out _, out var input);
        foreach (int batch in new[] { 1, 8 })
        {
            using var m = Compile(g, maxTokens: 8);
            Feed(m, input, (long)batch * 3 * 224 * 224);
            var (ms, rep) = Profile(m, batch, iters: 10);
            Console.WriteLine($"BENCH resnet50 batch {batch}: GPU {ms:F2} ms\n{rep}");
        }
    }
}
