using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Submission budget (<see cref="DeviceInfo.SubmissionBudgetMs"/>): with a tiny budget every launch becomes its own submission;
/// results must still match the CPU oracle, across the calibration run, plan replay and different token counts.
/// </summary>
public sealed class SubmissionBudgetTests
{
    [Theory]
    [InlineData("0.001")]   // every launch alone
    [InlineData("1")]       // a few launches per submission
    [InlineData("0")]       // budget off (single submission)
    public void ChunkedExecutionMatchesCpu(string budgetMs)
    {
        Gpu.Require();
        var provider = new VulkanProvider();
        var info = provider.GetDevices().First(d => d.Name == Gpu.Vulkan!.Info.Name);
        var opts = new DeviceOptions();
        opts.Settings["submission_budget_ms"] = budgetMs;
        using var dev = provider.OpenDevice(info, opts);

        const int H = 64, F = 96, Hq = 4, Hkv = 2, D = 16;
        var tg = new TG();
        var T = TG.T;
        var x = tg.InF("x", T, H);
        var w1 = tg.RandW("w1", DataType.F32, 0.3f, F, H);
        var w2 = tg.RandW("w2", DataType.F32, 0.3f, H, F);
        var w3 = tg.RandW("w3", DataType.F32, 0.3f, Hkv * D, H);
        var q = tg.InF("q", T, Hq, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        var h = tg.B.Add(tg.B.MatMul(tg.B.Silu(tg.B.MatMul(x, w1)), w2), x);
        var kv = tg.B.Reshape(tg.B.MatMul(h, w3), 0, Hkv, D);
        tg.B.CacheWrite(kc, kv); tg.B.CacheWrite(vc, kv);
        tg.B.Output("h", h);
        tg.B.Output("o", tg.B.CachedAttention(q, kc, vc, 0.25f));
        var g = tg.B.Build();

        CompiledModel Compile(IComputeDevice d) => CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = d }],
            Compile = new CompileOptions { MaxTokens = 32, ContextLength = 128, SymbolBounds = new Dictionary<string, long> { ["R"] = 32 } },
            ReservePerDevice = 0,
        });
        using var vk = Compile(dev);
        using var cpu = Compile(Gpu.Cpu);
        int past = 0;
        foreach (int t in new[] { 20, 20, 7, 1, 1, 20 })   // repeated token counts replay a plan
        {
            var xs = Rnd.F(t * H, 10 + past);
            var qs = Rnd.F(t * Hq * D, 20 + past);
            foreach (var m in new[] { vk, cpu })
            {
                m.SetInput(x, MemoryMarshal.AsBytes(xs.AsSpan())); m.SetInput(q, MemoryMarshal.AsBytes(qs.AsSpan()));
                var p = new RunParameters();
                p.Symbols["T"] = t; p.Symbols["R"] = t; p.Scalars["n_past"] = past;
                m.Run(p);
            }
            foreach (var (name, v) in g.Outputs)
            {
                long n = t * (name == "h" ? H : Hq * D);
                var a = new float[n];
                var b = new float[n];
                vk.GetOutput<float>(name, a); cpu.GetOutput<float>(name, b);
                for (long i = 0; i < n; i++)
                {
                    Assert.True(Math.Abs(a[i] - b[i]) <= 3e-3 + 3e-3 * Math.Abs(b[i]), $"{name}[{i}] T={t} past={past}: {a[i]} vs {b[i]}");
                }
            }
            past += t;
        }
    }
}
