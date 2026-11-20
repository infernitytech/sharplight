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

    private static ComputeGraph BuildGraph(out Value x, out Value ids, out Value slotWeights, List<NativeSource>? mapped = null, int experts = Experts)
    {
        var tg = new TG();
        // Expert banks in native memory (as a mapped file's) when `mapped` collects their sources: the sparse GPU compilation copies single experts.
        Value ExpertBank(string name, float[] data, params long[] dims)
        {
            if (mapped is null)
            {
                return tg.WF(name, data, dims);
            }

            var source = new NativeSource(data.Length * sizeof(float), (ptr, bytes) => System.Runtime.InteropServices.Marshal.Copy(data, 0, ptr, data.Length));
            mapped.Add(source);
            tg.Weights[name] = new Models.WeightInfo { Name = name, Type = DataType.F32, Shape = new Shape([.. dims.Select(d => (Dim)d)]), Source = source };
            return tg.B.Weight(name);
        }

        x = tg.InF("x", TG.T, Hidden);
        ids = tg.In("ids", DataType.I32, TG.T, Slots);
        slotWeights = tg.InF("slot_weights", TG.T, Slots);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Rnd.F(Hidden * Hidden, layer, 0.1f), Hidden, Hidden));
            var slotWeightRows = tg.B.Reshape(slotWeights, 0, Slots, 1);
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
            var graph = BuildGraph(out var x, out var ids, out var slotWeights, mapped, experts);
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
            var rowIds = Rnd.I(rows * Slots, 5, experts);
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
}
