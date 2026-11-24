using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Routed experts placed on the CPU fallback with a streamed GPU compilation (<see cref="PlacementOptions.StreamExpertsMinRows"/>): large
/// batches run the experts on the GPU with their weights copied in and match the GPU-only compilation, small ones run on the CPU and match
/// the CPU-only compilation.
/// </summary>
public sealed class StreamedExpertsTests
{
    private const int Hidden = 64;
    private const int Experts = 8;
    private const int Slots = 2;
    private const int Inner = 128;
    private const int Layers = 3;
    private const long DenseBytes = Hidden * Hidden * sizeof(float);
    private const long ExpertBytes = 2L * Experts * Inner * Hidden * sizeof(float);
    private const int MinRows = 16;

    private static ComputeGraph BuildGraph(out Value x, out Value ids, out Value slotWeights)
    {
        var tg = new TG();
        x = tg.InF("x", TG.T, Hidden);
        ids = tg.In("ids", DataType.I32, TG.T, Slots);
        slotWeights = tg.InF("slot_weights", TG.T, Slots);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Rnd.F(Hidden * Hidden, layer, 0.1f), Hidden, Hidden));
            var slotWeightRows = tg.B.Reshape(slotWeights, 0, Slots, 1);
            var up = tg.B.Emit1("MatMulId", [h, tg.WF($"up{layer}", Rnd.F(Experts * Inner * Hidden, 10 + layer, 0.1f), Experts, Inner, Hidden), ids]);
            var act = tg.B.Mul(up, up);
            var down = tg.B.Emit1("MatMulId", [act, tg.WF($"down{layer}", Rnd.F(Experts * Hidden * Inner, 20 + layer, 0.1f), Experts, Hidden, Inner), ids]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        return tg.B.Build();
    }

    /// <summary>
    /// The layers of <see cref="BuildGraph"/> with a router of their own (a matmul, the top <see cref="Slots"/> experts, a softmax over them), so the
    /// expert ids and weights are values the GPU computes, and the expert banks in native memory (collected into <paramref name="mapped"/>).
    /// </summary>
    private static ComputeGraph BuildRoutedGraph(out Value x, List<NativeSource> mapped, int experts)
    {
        var tg = new TG();
        Value ExpertBank(string name, float[] data, params long[] dims)
        {
            var source = new NativeSource(data.Length * sizeof(float), (ptr, bytes) => System.Runtime.InteropServices.Marshal.Copy(data, 0, ptr, data.Length));
            mapped.Add(source);
            tg.Weights[name] = new Models.WeightInfo { Name = name, Type = DataType.F32, Shape = new Shape([.. dims.Select(d => (Dim)d)]), Source = source };
            return tg.B.Weight(name);
        }

        x = tg.InF("x", TG.T, Hidden);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Rnd.F(Hidden * Hidden, layer, 0.1f), Hidden, Hidden));
            var logits = tg.B.MatMul(h, tg.WF($"router{layer}", Rnd.F(experts * Hidden, 30 + layer, 1f), experts, Hidden));
            var top = tg.B.Emit("TopK", [logits], Attributes.From(("k", (long)Slots)));
            var ids = top[1];
            var slotWeightRows = tg.B.Reshape(tg.B.Emit1("Softmax", [top[0]]), 0, Slots, 1);
            var up = tg.B.Emit1("MatMulId", [h, ExpertBank($"up{layer}", Rnd.F(experts * Inner * Hidden, 10 + layer, 0.1f), experts, Inner, Hidden), ids]);
            var act = tg.B.Mul(up, up);
            var down = tg.B.Emit1("MatMulId", [act, ExpertBank($"down{layer}", Rnd.F(experts * Hidden * Inner, 20 + layer, 0.1f), experts, Hidden, Inner), ids]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        return tg.B.Build();
    }

    private static CompileOptions Compile => new()
    {
        MaxTokens = 64,
        ContextLength = 64,
        DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot, DeviceFeatures.MatrixF16, DeviceFeatures.MatrixF16Small, DeviceFeatures.MatrixBF16, DeviceFeatures.LibraryGemm },
    };

    /// <summary>
    /// Small passes run the CPU layers' experts split: the GPU copies in the experts of its share of the slots (sparse compilation), the CPU
    /// computes the other slots; the sum matches the CPU-only compilation, whatever share the GPU takes.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void SplitExpertsMatchTheCpu(int rows)
    {
        Gpu.Require();
        var mapped = new List<NativeSource>();
        try
        {
            // Enough experts that a sparse pass copies few of them (four rows of two slots select at most a quarter of 32).
            const int experts = 32;
            const long expertBytes = 2L * experts * Inner * Hidden * sizeof(float);
            var graph = BuildRoutedGraph(out var x, mapped, experts);
            var split = new PlacementOptions
            {
                // The dense layers, the streaming memory, the sparse banks (a quarter layer) and a quarter layer more: some experts move to the CPU.
                Devices = [new DeviceSlot { Device = Gpu.Device!, MemoryLimit = Layers * DenseBytes + expertBytes + expertBytes / 2 + (128 << 10) }],
                Fallback = new DeviceSlot { Device = Gpu.Cpu },
                Compile = Compile,
                ReservePerDevice = 0,
                ReservePerSegment = 0,
                StreamExpertsMinRows = MinRows,
            };
            using var model = CompiledModel.Compile(graph, split);
            if (model.SparseSegmentCount == 0)
            {
                Assert.Skip("The device has no sparse expert compilation.");
            }

            using var cpu = CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = Gpu.Cpu }], Compile = Compile });
            var input = Rnd.F(rows * Hidden, 3);
            float[] Run(CompiledModel m)
            {
                m.SetInput(x, System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
                var p = new RunParameters();
                p.Symbols["T"] = rows;
                m.Run(p);
                var y = new float[rows * Hidden];
                m.GetOutput<float>("y", y);
                return y;
            }

            var expected = Run(cpu);
            // Several passes: the GPU's share of the slots moves between them, and each pass copies its experts over the previous ones.
            for (int pass = 0; pass < 4; pass++)
            {
                var actual = Run(model);
                for (int i = 0; i < actual.Length; i++)
                {
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3 + 1e-3 * Math.Abs(expected[i]), $"pass {pass} [{i}]: {actual[i]} vs the CPU's {expected[i]}");
                }
            }
        }
        finally
        {
            foreach (var source in mapped)
            {
                source.Dispose();
            }
        }
    }

    /// <summary>
    /// Large passes over routed CPU layers copy only the experts they select into the streaming memory (the rest of the memory keeps the
    /// experts of earlier layers and passes) and match the GPU-only compilation, pass after pass with different routings.
    /// </summary>
    [Fact]
    public void StreamedSelectedExpertsMatchTheGpu()
    {
        Gpu.Require();
        var mapped = new List<NativeSource>();
        try
        {
            // Twenty rows of two slots select at most 40 of 96 experts.
            const int experts = 96;
            const int rows = 20;
            const long expertBytes = 2L * experts * Inner * Hidden * sizeof(float);
            const long routerBytes = experts * Hidden * sizeof(float);
            var graph = BuildRoutedGraph(out var x, mapped, experts);
            var split = new PlacementOptions
            {
                // The dense layers and routers, the streaming memory, the sparse banks and half a layer more: most experts move to the CPU.
                Devices = [new DeviceSlot { Device = Gpu.Device!, MemoryLimit = Layers * (DenseBytes + routerBytes) + expertBytes + expertBytes / 2 + (128 << 10) }],
                Fallback = new DeviceSlot { Device = Gpu.Cpu },
                Compile = Compile,
                ReservePerDevice = 0,
                ReservePerSegment = 0,
                StreamExpertsMinRows = MinRows,
            };
            using var model = CompiledModel.Compile(graph, split);
            Assert.True(model.StreamedSegmentCount >= 2, $"{model.StreamedSegmentCount} streamed segments");
            using var gpu = CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = Gpu.Device! }], Compile = Compile });
            float[] Run(CompiledModel m, float[] input)
            {
                m.SetInput(x, System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
                var p = new RunParameters();
                p.Symbols["T"] = rows;
                m.Run(p);
                var y = new float[rows * Hidden];
                m.GetOutput<float>("y", y);
                return y;
            }

            for (int pass = 0; pass < 4; pass++)
            {
                var input = Rnd.F(rows * Hidden, 3 + pass);
                var expected = Run(gpu, input);
                var actual = Run(model, input);
                for (int i = 0; i < actual.Length; i++)
                {
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 + 1e-4 * Math.Abs(expected[i]), $"pass {pass} [{i}]: {actual[i]} vs the GPU's {expected[i]}");
                }
            }
        }
        finally
        {
            foreach (var source in mapped)
            {
                source.Dispose();
            }
        }
    }

    [Theory]
    [InlineData(40, 1)]
    [InlineData(40, 2)]
    [InlineData(3, 1)]
    public void StreamedAndCpuExpertsMatchTheCpu(int rows, int buffers)
    {
        Gpu.Require();
        var graph = BuildGraph(out var x, out var ids, out var slotWeights);
        var split = new PlacementOptions
        {
            // The dense layers, a quarter layer of experts beyond the streaming memories and the transfer buffers: some experts move to the CPU.
            Devices = [new DeviceSlot { Device = Gpu.Device!, MemoryLimit = Layers * DenseBytes + buffers * ExpertBytes + ExpertBytes / 4 + (128 << 10) }],
            Fallback = new DeviceSlot { Device = Gpu.Cpu },
            Compile = Compile,
            ReservePerDevice = 0,
        ReservePerSegment = 0,
            StreamExpertsMinRows = MinRows,
            StreamBuffers = buffers,
        };
        using var model = CompiledModel.Compile(graph, split);
        Assert.Contains(model.Plan.GroupDevice, kv => kv.Key.EndsWith(PlacementPlanner.ExpertGroupSuffix, StringComparison.Ordinal) && kv.Value == 1);
        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(0, model.Plan.GroupDevice[$"layers.{l}"]));
        Assert.True(model.StreamedSegmentCount > 0);
        using var cpu = CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = Gpu.Cpu }], Compile = Compile });

        var input = Rnd.F(rows * Hidden, 3);
        var rowIds = Rnd.I(rows * Slots, 5, Experts);
        var weights = Rnd.Pos(rows * Slots, 7, 0.1f, 1f);
        float[] Run(CompiledModel m)
        {
            m.SetInput(x, System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
            m.SetInput(ids, System.Runtime.InteropServices.MemoryMarshal.AsBytes(rowIds.AsSpan()));
            m.SetInput(slotWeights, System.Runtime.InteropServices.MemoryMarshal.AsBytes(weights.AsSpan()));
            var p = new RunParameters();
            p.Symbols["T"] = rows;
            m.Run(p);
            var y = new float[rows * Hidden];
            m.GetOutput<float>("y", y);
            return y;
        }

        // The whole graph on the GPU: the streamed experts run the same kernels, so they match it closely; the GPU's grouped expert
        // products may round to f16 above 16 items, so the CPU comparison is looser there.
        using var gpu = CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = Gpu.Device! }], Compile = Compile });
        var cpuResult = Run(cpu);
        var gpuResult = Run(gpu);
        // Twice: the second pass copies the streamed weights into buffers the first one already used (with two memories, the second layer's
        // copy starts while the first runs).
        Run(model);
        var actual = Run(model);
        bool streamed = rows >= MinRows;
        for (int i = 0; i < actual.Length; i++)
        {
            float expected = streamed ? gpuResult[i] : cpuResult[i];
            Assert.True(Math.Abs(expected - actual[i]) <= 1e-4 + 1e-4 * Math.Abs(expected), $"rows={rows} [{i}]: {actual[i]} vs {expected}");
            Assert.True(Math.Abs(cpuResult[i] - actual[i]) <= 3e-2 + 3e-2 * Math.Abs(cpuResult[i]), $"rows={rows} [{i}]: {actual[i]} vs the CPU's {cpuResult[i]}");
        }
    }

    /// <summary>
    /// Routed experts on the CPU left on disk (<see cref="CompileOptions.DiskExperts"/>, the smallest cache): the router runs on the GPU, the CPU
    /// reads the selected experts from the file, and the outputs equal those of the same placement with resident experts (CPU experts for
    /// every pass: no streamed compilations, which the disk placement does not make either), for decode-sized and prefill-sized passes. On a
    /// device that copies from the expert cache (<see cref="IComputeDevice.CopiesFromExpertCache"/>) small passes split with the GPU, which
    /// copies its experts from the page-locked cache slots: those match the CPU as the split of resident experts does.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(40)]
    public void DiskExpertsMatchTheResidentExperts(int rows)
    {
        Gpu.Require();
        const int experts = 96;
        using var file = new Sharplight.Tests.FileWeights();
        var tg = new TG();
        Value ExpertBank(string name, float[] data, params long[] dims)
        {
            tg.Weights[name] = file.AddF32(name, data, dims);
            return tg.B.Weight(name);
        }

        var x = tg.InF("x", TG.T, Hidden);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Rnd.F(Hidden * Hidden, layer, 0.1f), Hidden, Hidden));
            var logits = tg.B.MatMul(h, tg.WF($"router{layer}", Rnd.F(experts * Hidden, 30 + layer, 1f), experts, Hidden));
            var top = tg.B.Emit("TopK", [logits], Attributes.From(("k", (long)Slots)));
            var ids = top[1];
            var slotWeightRows = tg.B.Reshape(tg.B.Emit1("Softmax", [top[0]]), 0, Slots, 1);
            var gate = tg.B.Emit1("MatMulId", [h, ExpertBank($"gate{layer}", Rnd.F(experts * Inner * Hidden, 10 + layer, 0.1f), experts, Inner, Hidden), ids]);
            var up = tg.B.Emit1("MatMulId", [h, ExpertBank($"up{layer}", Rnd.F(experts * Inner * Hidden, 40 + layer, 0.1f), experts, Inner, Hidden), ids]);
            var act = tg.B.Mul(tg.B.Emit1("Silu", [gate]), up);
            var down = tg.B.Emit1("MatMulId", [act, ExpertBank($"down{layer}", Rnd.F(experts * Hidden * Inner, 20 + layer, 0.1f), experts, Hidden, Inner), ids]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        var graph = tg.B.Build();
        using var store = new Sharplight.Runtime.Experts.ExpertStore(cacheBytes: 1, Sharplight.Runtime.Experts.ExpertReaders.CreateThreads(4));
        PlacementOptions Placement(Sharplight.Runtime.Experts.ExpertStore? disk) => new()
        {
            // The dense layers and routers, the transfer buffers and the sparse banks (3 x 8 experts of 32 KiB) only: every layer's experts go to the CPU.
            Devices = [new DeviceSlot { Device = Gpu.Device!, MemoryLimit = Layers * (DenseBytes + experts * Hidden * sizeof(float)) + (1280 << 10) }],
            Fallback = new DeviceSlot { Device = Gpu.Cpu },
            Compile = new CompileOptions { MaxTokens = 64, ContextLength = 64, DisabledFeatures = Compile.DisabledFeatures, DiskExperts = disk },
            ReservePerDevice = 0,
            ReservePerSegment = 0,
            StreamExpertsMinRows = disk is null ? 0 : MinRows,
        };

        using var resident = CompiledModel.Compile(graph, Placement(null));
        using var disk = CompiledModel.Compile(graph, Placement(store));
        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(1, disk.Plan.GroupDevice[$"layers.{l}{PlacementPlanner.ExpertGroupSuffix}"]));
        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(0, disk.Plan.GroupDevice[$"layers.{l}"]));
        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(0, resident.Plan.GroupDevice[$"layers.{l}"]));
        Assert.Equal(Layers, disk.Plan.DiskExpertGroups.Count);
        Assert.Equal(0, disk.StreamedSegmentCount);
        bool split = Gpu.Device!.CopiesFromExpertCache && Gpu.Device.StreamsSelectedExperts;
        Assert.Equal(split, disk.SparseSegmentCount > 0);

        var input = Rnd.F(rows * Hidden, 3);
        float[] Run(CompiledModel m)
        {
            m.SetInput(x, System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
            var p = new RunParameters();
            p.Symbols["T"] = rows;
            m.Run(p);
            var y = new float[rows * Hidden];
            m.GetOutput<float>("y", y);
            return y;
        }

        var expected = Run(resident);
        // Several passes: a split pass's GPU share moves between them (the balancer), its experts copied from the cache each time.
        for (int pass = 0; pass < 4; pass++)
        {
            var actual = Run(disk);
            if (!split || rows > 4)
            {
                Assert.Equal(expected, actual);
                continue;
            }

            for (int i = 0; i < actual.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3 + 1e-3 * Math.Abs(expected[i]), $"pass {pass} [{i}]: {actual[i]} vs the CPU's {expected[i]}");
            }
        }

        var stats = store.Statistics;
        Assert.True(stats.Misses > 0 && stats.Hits + stats.ReadAhead > 0);
    }
}
