namespace Sharplight.Cpu.Tests;

/// <summary>The spinning worker pool behind <see cref="CpuParallel.For"/>: coverage, exceptions, nesting and concurrent callers.</summary>
public class CpuParallelTests
{
    [Theory]
    [InlineData(1), InlineData(2), InlineData(7), InlineData(64), InlineData(1000)]
    public void EveryIndexRunsOnce(int n)
    {
        for (int repeat = 0; repeat < 50; repeat++)
        {
            var hits = new int[n];
            CpuParallel.For(n, i => Interlocked.Increment(ref hits[i]));
            Assert.All(hits, h => Assert.Equal(1, h));
        }
    }

    [Fact]
    public void ExceptionPropagatesAndPoolRecovers()
    {
        var ex = Assert.ThrowsAny<Exception>(() => CpuParallel.For(100, i =>
        {
            if (i == 37)
            {
                throw new InvalidOperationException("boom");
            }
        }));
        Assert.Contains("boom", ex is AggregateException a ? a.InnerExceptions[0].Message : ex.Message);
        var hits = new int[100];
        CpuParallel.For(100, i => Interlocked.Increment(ref hits[i]));
        Assert.All(hits, h => Assert.Equal(1, h));
    }

    [Fact]
    public void NestedCallsComplete()
    {
        var hits = new int[16 * 16];
        CpuParallel.For(16, i => CpuParallel.For(16, j => Interlocked.Increment(ref hits[i * 16 + j])));
        Assert.All(hits, h => Assert.Equal(1, h));
    }

    [Fact]
    public void ConcurrentCallersComplete()
    {
        var hits = new int[8, 200];
        Parallel.For(0, 8, c =>
        {
            for (int repeat = 0; repeat < 20; repeat++)
            {
                CpuParallel.For(10, i => Interlocked.Increment(ref hits[c, repeat * 10 + i]));
            }
        });
        foreach (int h in hits)
        {
            Assert.Equal(1, h);
        }
    }
}
