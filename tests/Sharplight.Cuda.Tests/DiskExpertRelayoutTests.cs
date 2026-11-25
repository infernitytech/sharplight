using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Runtime.Experts;
using Sharplight.Tests;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// Routed experts on disk on a CUDA placement: the GPU copies the experts of its share of a split pass from the cache's page-locked slots.
/// When a second compilation registers more experts with the same store (a placement retried after an out-of-memory failure put one more
/// layer on the CPU), the store lays its pools out again at the next pass. DeepSeek-V4-Flash with <c>--batch 64</c> hit that: the first pools
/// had been page-locked twice and were released once, so they stayed registered after they were freed, and the new pools at the same
/// addresses counted as registered with the freed pages. The GPU then copied stale experts (wrong text) or failed the copy. The output
/// must match the CPU's with the experts resident.
/// </summary>
public sealed class DiskExpertRelayoutTests
{
    private const int Hidden = 128;
    private const int Inner = 256;
    private const int Experts = 64;
    private const int Slots = 2;

    /// <summary>
    /// Bytes of the expert cache: smaller than two layers' experts (48 MiB), so they stay on disk, and larger than glibc's largest mmap
    /// threshold (32 MiB), so a pool freed and allocated again at the same size gets its own mapping, usually at the old address.
    /// </summary>
    private const long CacheBytes = 40L << 20;

    /// <summary>A model of <paramref name="layers"/> MoE layers (a dense projection, a router, gated experts, the weighted slot sum) over the weights of <paramref name="file"/>.</summary>
    private static ComputeGraph BuildGraph(FileWeights file, Dictionary<string, Models.WeightInfo> weights, int layers, out Value x, out long denseBytes)
    {
        var tg = new TG();
        Value Weight(string name, int seed, float scale, params long[] dims)
        {
            if (!weights.TryGetValue(name, out var info))
            {
                info = file.AddF32(name, Rnd.F(dims.Aggregate(1L, (a, b) => a * b), seed, scale), dims);
                weights[name] = info;
            }

            tg.Weights[name] = info;
            return tg.B.Weight(name);
        }

        x = tg.InF("x", TG.T, Hidden);
        var h = x;
        for (int layer = 0; layer < layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, Weight($"dense{layer}", layer, 0.08f, Hidden, Hidden));
            var logits = tg.B.MatMul(h, Weight($"router{layer}", 100 + layer, 1f, Experts, Hidden));
            var top = tg.B.Emit("TopK", [logits], Attributes.From(("k", (long)Slots)));
            var slotWeightRows = tg.B.Reshape(tg.B.Emit1("Softmax", [top[0]]), 0, Slots, 1);
            var gate = tg.B.Emit1("MatMulId", [h, Weight($"gate{layer}", 200 + layer, 0.08f, Experts, Inner, Hidden), top[1]]);
            var up = tg.B.Emit1("MatMulId", [h, Weight($"up{layer}", 300 + layer, 0.08f, Experts, Inner, Hidden), top[1]]);
            var act = tg.B.Mul(tg.B.Emit1("Silu", [gate]), up);
            var down = tg.B.Emit1("MatMulId", [act, Weight($"down{layer}", 400 + layer, 0.06f, Experts, Hidden, Inner), top[1]]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        denseBytes = layers * (Hidden + Experts) * Hidden * (long)sizeof(float);
        return tg.B.Build();
    }

    /// <summary>The dense layers on the GPU, every layer's experts on the CPU (the GPU's limit has no room for one layer's 24 MiB).</summary>
    private static PlacementOptions Options(IComputeDevice gpu, IComputeDevice cpu, long denseBytes, ExpertStore? store) => new()
    {
        Devices = [new DeviceSlot { Device = gpu, MemoryLimit = denseBytes + (8L << 20) }],
        Fallback = new DeviceSlot { Device = cpu },
        Compile = new CompileOptions { MaxTokens = 16, ContextLength = 16, DiskExperts = store },
        ReservePerDevice = 0,
        ReservePerSegment = 0,
    };

    private static float[] Run(CompiledModel model, Value x, float[] input, int rows)
    {
        model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan()));
        var parameters = new RunParameters();
        parameters.Symbols["T"] = rows;
        model.Run(parameters);
        var y = new float[rows * Hidden];
        model.GetOutput<float>("y", y);
        return y;
    }

    [Fact]
    public void SplitPassesAfterTheCacheIsLaidOutAgainMatchTheCpu()
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        using var gpu = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions { UseCompilationCache = false });
        var cpuProvider = new CpuExecutionProvider();
        using var cpu = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]);
        using var file = new FileWeights();
        var weights = new Dictionary<string, Models.WeightInfo>();
        using var store = new ExpertStore(CacheBytes, ExpertReaders.CreateThreads(4));

        // The first compilation registers two layers' experts and page-locks the pools for the GPU.
        var twoLayers = BuildGraph(file, weights, 2, out var x2, out long dense2);
        using (var first = CompiledModel.Compile(twoLayers, Options(gpu, cpu, dense2, store)))
        {
            Assert.Equal(2, first.Plan.DiskExpertGroups.Count);
            Assert.True(first.SparseSegmentCount > 0, "no split passes: the test would not reach the GPU's copies from the cache");
            Run(first, x2, Rnd.F(Hidden, 1, 1f), 1);
        }

        // The second registers a third layer: the pools are laid out again at its first pass.
        var threeLayers = BuildGraph(file, weights, 3, out var x, out long dense3);
        using var model = CompiledModel.Compile(threeLayers, Options(gpu, cpu, dense3, store));
        Assert.Equal(3, model.Plan.DiskExpertGroups.Count);
        Assert.True(model.SparseSegmentCount > 0);
        using var reference = CompiledModel.Compile(threeLayers, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = cpu }],
            Compile = new CompileOptions { MaxTokens = 16, ContextLength = 16 },
        });

        foreach (int rows in new[] { 1, 3, 1, 2, 1, 1 })
        {
            var input = Rnd.F(rows * Hidden, 10 + rows, 1f);
            var expected = Run(reference, x, input, rows);
            var actual = Run(model, x, input, rows);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1 + Math.Abs(expected[i])), $"{rows} rows, value {i}: {actual[i]} instead of {expected[i]}");
            }
        }
    }
}
