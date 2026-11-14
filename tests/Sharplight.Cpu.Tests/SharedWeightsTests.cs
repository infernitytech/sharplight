using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

/// <summary><see cref="SharedWeights"/>: the registry contract and weight memory shared by two CPU compilations.</summary>
public class SharedWeightsTests
{
    private sealed class Handle
    {
        public bool Freed;
    }

    [Fact]
    public void Acquire_SharesOneBuffer_FreesWithLastHolder()
    {
        var shared = new SharedWeights();
        var tg = new TestGraph();
        tg.WF("w", [1, 2, 3, 4], 4);
        var key = WeightBufferKey.For(tg.Weights["w"]);
        var device = new object();
        int allocations = 0;
        var first = shared.Acquire(device, key, 16, () => { allocations++; return new Handle(); }, h => h.Freed = true);
        var second = shared.Acquire(device, key, 16, () => { allocations++; return new Handle(); }, h => h.Freed = true);
        var otherLayout = shared.Acquire(device, key with { Layout = "x" }, 16, () => { allocations++; return new Handle(); }, h => h.Freed = true);
        Assert.Same(first.Buffer, second.Buffer);
        Assert.True(first.MustFill);
        Assert.False(second.MustFill);
        Assert.True(otherLayout.MustFill);
        Assert.Equal(2, allocations);
        Assert.Equal(32, shared.ResidentBytes);
        Assert.True(shared.IsResident(tg.Weights["w"]));

        first.Filled();
        second.WaitFilled();
        var buffer = first.Buffer;
        first.Dispose();
        Assert.False(buffer.Freed);
        second.Dispose();
        Assert.True(buffer.Freed);
        otherLayout.Dispose();
        Assert.Equal(0, shared.ResidentBytes);
        Assert.False(shared.IsResident(tg.Weights["w"]));
    }

    [Fact]
    public void FailedFill_ReachesWaiters_AndIsNotReused()
    {
        var shared = new SharedWeights();
        var tg = new TestGraph();
        tg.WF("w", [1, 2, 3, 4], 4);
        var key = WeightBufferKey.For(tg.Weights["w"]);
        var device = new object();
        var filler = shared.Acquire(device, key, 16, () => new Handle(), _ => { });
        var waiter = shared.Acquire(device, key, 16, () => new Handle(), _ => { });
        filler.Dispose();
        Assert.Throws<InvalidOperationException>(waiter.WaitFilled);
        using var fresh = shared.Acquire(device, key, 16, () => new Handle(), _ => { });
        Assert.True(fresh.MustFill);
        waiter.Dispose();
    }

    [Fact]
    public void TwoCompilations_ShareCopiedWeights()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4);
        // Not zero-copy: the CPU provider copies the bytes, which the second compilation then reuses.
        var bias = tg.W("b", Tensors.DataType.F32, new Tensors.Shape(4), System.Runtime.InteropServices.MemoryMarshal.AsBytes(new float[] { 1, 2, 3, 4 }.AsSpan()).ToArray(), zeroCopy: false);
        tg.B.Output("y", tg.B.Add(x, bias, "y"));
        var graph = tg.B.Build();
        var shared = new SharedWeights();
        CompiledModel Compile() => CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Kit.Device }],
            Compile = new CompileOptions { MaxTokens = 4, ContextLength = 4, SharedWeights = shared },
            ReservePerDevice = 0,
        });

        Assert.True(Kit.Device.SharesWeights);
        using var a = Compile();
        Assert.Equal(16, shared.ResidentBytes);
        using (var b = Compile())
        {
            Assert.Equal(16, shared.ResidentBytes);
            foreach (var m in new[] { a, b })
            {
                Kit.Set(m, x, new float[] { 10, 20, 30, 40, 50, 60, 70, 80 });
                var p = Kit.P(2);
                m.Run(p);
                Assert.Equal([11f, 22, 33, 44, 51, 62, 73, 84], Kit.Out(m, "y", p));
            }
        }

        Assert.Equal(16, shared.ResidentBytes);
    }
}
