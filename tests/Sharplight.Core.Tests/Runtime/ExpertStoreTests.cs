using Sharplight.Models;
using Sharplight.Runtime.Experts;
using Sharplight.Tests;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>
/// The cache of routed experts left on disk (<see cref="ExpertStore"/>): an acquired expert's bytes are the stored ones, repeated experts
/// hit, the least recently used expert no pass holds is evicted, held experts never are, prefetches count as hits, and the decision to
/// leave experts on disk follows the host budget and the cache asked for.
/// </summary>
public sealed unsafe class ExpertStoreTests
{
    private const int Experts = 200;
    private const int ExpertFloats = 3000;   // 12000 bytes: experts straddle page boundaries

    private static (FileWeights File, WeightInfo Gate, WeightInfo Down, float[] GateData, float[] DownData) Weights()
    {
        var file = new FileWeights();
        var gateData = Enumerable.Range(0, Experts * ExpertFloats).Select(i => (float)i).ToArray();
        var downData = Enumerable.Range(0, Experts * ExpertFloats).Select(i => -(float)i).ToArray();
        var gate = file.AddF32("blk.0.ffn_gate_exps.weight", gateData, Experts, 30, 100);
        var down = file.AddF32("blk.0.ffn_down_exps.weight", downData, Experts, 100, 30);
        return (file, gate, down, gateData, downData);
    }

    /// <summary>A store whose cache holds 64 slots of the test's experts (the smallest pool).</summary>
    private static ExpertStore Store() => new(cacheBytes: 1, ExpertReaders.CreateThreads(4));

    private static void AssertExpert(byte* pointer, float[] data, int expert)
        => Assert.True(new ReadOnlySpan<float>(pointer, ExpertFloats).SequenceEqual(data.AsSpan(expert * ExpertFloats, ExpertFloats)), $"expert {expert}");

    [Fact]
    public void AcquiredExpertsHoldTheirStoredBytes_AndRepeatsHit()
    {
        var (file, gate, down, gateData, downData) = Weights();
        using var _ = file;
        using var store = Store();
        Assert.True(store.TryRegister(gate, "blk.0#experts", 2));
        Assert.True(store.TryRegister(down, "blk.0#experts", 2));
        var gateTensor = store.Find(gate)!;
        var downTensor = store.Find(down)!;
        var pointers = new nint[Experts];
        var pointers2 = new nint[Experts];
        int[] experts = [3, 17, 199, 0];
        fixed (nint* p = pointers)
        fixed (nint* q = pointers2)
        {
            store.Acquire(gateTensor, null, experts, (byte**)p, (byte**)q, prefetchLayer: true);
            foreach (int e in experts)
            {
                AssertExpert((byte*)pointers[e], gateData, e);
            }

            store.Release(gateTensor, null, experts);
            var first = store.Statistics;
            Assert.Equal(4, first.Misses);
            Assert.Equal(0, first.Hits);
            Assert.Equal(4, first.Prefetched);   // the layer's down projection, read ahead

            store.Acquire(downTensor, null, experts, (byte**)p, (byte**)q, prefetchLayer: false);
            foreach (int e in experts)
            {
                AssertExpert((byte*)pointers[e], downData, e);
            }

            store.Release(downTensor, null, experts);
            store.Acquire(gateTensor, null, [17, 5], (byte**)p, (byte**)q, prefetchLayer: false);
            AssertExpert((byte*)pointers[17], gateData, 17);
            AssertExpert((byte*)pointers[5], gateData, 5);
            store.Release(gateTensor, null, [17, 5]);
        }

        var stats = store.Statistics;
        Assert.Equal(5, stats.Misses);
        Assert.Equal(5, stats.Hits);
        Assert.True(stats.BytesRead >= 9 * ExpertFloats * sizeof(float));
    }

    [Fact]
    public void TheLeastRecentlyUsedExpertIsEvicted_HeldOnesAreNot()
    {
        var (file, gate, _, gateData, _) = Weights();
        using var __ = file;
        using var store = Store();
        store.TryRegister(gate, "blk.0#experts", 2);
        var tensor = store.Find(gate)!;
        var pointers = new nint[Experts];
        fixed (nint* p = pointers)
        {
            int slots = store.Statistics.CacheSlots;
            Assert.Equal(0, slots);   // laid out on first use
            int chunk = store.ChunkExperts(tensor);
            slots = store.Statistics.CacheSlots;
            Assert.Equal(64, slots);
            Assert.Equal(16, chunk);

            // Expert 0 stays held while every other slot cycles through other experts.
            store.Acquire(tensor, null, [0], (byte**)p, null, false);
            var held = pointers[0];
            for (int e = 1; e < Experts; e++)
            {
                store.Acquire(tensor, null, [e], (byte**)p, null, false);
                store.Release(tensor, null, [e]);
            }

            AssertExpert((byte*)held, gateData, 0);
            store.Release(tensor, null, [0]);
            store.ResetStatistics();

            // The most recent 63 experts are cached, the older ones were evicted.
            store.Acquire(tensor, null, [Experts - 1, Experts - 63], (byte**)p, null, false);
            store.Release(tensor, null, [Experts - 1, Experts - 63]);
            Assert.Equal(2, store.Statistics.Hits);
            store.Acquire(tensor, null, [1], (byte**)p, null, false);
            store.Release(tensor, null, [1]);
            Assert.Equal(1, store.Statistics.Misses);
            AssertExpert((byte*)pointers[1], gateData, 1);
        }
    }

    /// <summary>Experts used twice are protected: a pass over many new experts (a prefill) evicts the ones used once first.</summary>
    [Fact]
    public void ExpertsUsedTwiceSurviveAScanOfNewOnes()
    {
        var (file, gate, _, _, _) = Weights();
        using var __ = file;
        using var store = Store();
        store.TryRegister(gate, "blk.0#experts", 2);
        var tensor = store.Find(gate)!;
        var pointers = new nint[Experts];
        fixed (nint* p = pointers)
        {
            int[] favourites = [7, 8, 9, 10];
            for (int pass = 0; pass < 2; pass++)
            {
                store.Acquire(tensor, null, favourites, (byte**)p, null, false);
                store.Release(tensor, null, favourites);
            }

            for (int e = 20; e < Experts; e += 8)
            {
                var scan = Enumerable.Range(e, Math.Min(8, Experts - e)).ToArray();
                store.Acquire(tensor, null, scan, (byte**)p, null, false);
                store.Release(tensor, null, scan);
            }

            store.ResetStatistics();
            store.Acquire(tensor, null, favourites, (byte**)p, null, false);
            store.Release(tensor, null, favourites);
            Assert.Equal(4, store.Statistics.Hits);
        }
    }

    [Fact]
    public void MoreExpertsThanFreeSlotsAtOnceFail_AndLeaveTheCacheUsable()
    {
        var (file, gate, _, gateData, _) = Weights();
        using var __ = file;
        using var store = Store();
        store.TryRegister(gate, "blk.0#experts", 2);
        var tensor = store.Find(gate)!;
        var pointers = new nint[Experts];
        fixed (nint* p = pointers)
        {
            var many = Enumerable.Range(0, 65).ToArray();
            nint table = (nint)p;
            Assert.Throws<InvalidOperationException>(() => store.Acquire(tensor, null, many, (byte**)table, null, false));
            var fits = Enumerable.Range(100, 64).ToArray();
            store.Acquire(tensor, null, fits, (byte**)p, null, false);
            foreach (int e in fits)
            {
                AssertExpert((byte*)pointers[e], gateData, e);
            }

            store.Release(tensor, null, fits);
        }
    }

    [Theory]
    [InlineData(null, 30L << 30, 10L << 30, 64L << 30, false)]   // fits with the reserve
    [InlineData(null, 60L << 30, 10L << 30, 64L << 30, true)]    // does not fit: the cache takes what is left
    [InlineData(8L << 30, 30L << 30, 10L << 30, 64L << 30, true)] // a cache smaller than the experts
    [InlineData(40L << 30, 30L << 30, 10L << 30, 64L << 30, false)]
    public void TheExpertsGoToDiskWhenTheyDoNotFitTheHostOrTheCacheAskedFor(long? cache, long experts, long other, long host, bool disk)
    {
        using var store = new ExpertStore(cache, ExpertReaders.CreateThreads(1));
        Assert.Equal(disk, store.Decide(experts, other, host));
        Assert.Equal(disk, store.IsActive);
        if (disk)
        {
            Assert.Equal(cache ?? host - other - ExpertStore.HostReserve, store.CacheBytes);
        }
    }
}
