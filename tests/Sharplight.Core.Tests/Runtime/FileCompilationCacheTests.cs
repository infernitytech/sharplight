using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>The persistent compilation cache: validated entries, atomic writes, LRU size cap, concurrent writers, environment switch.</summary>
public sealed class FileCompilationCacheTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("sharplight-cache-test-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void EntriesRoundTripAcrossInstances()
    {
        new FileCompilationCache(_directory).Put("kernel/a", new byte[] { 1, 2, 3 });
        var other = new FileCompilationCache(_directory);
        Assert.True(other.TryGet("kernel/a", out var data));
        Assert.Equal(new byte[] { 1, 2, 3 }, data.ToArray());
        Assert.False(other.TryGet("kernel/b", out _));
        Assert.Equal(1, other.Hits);
    }

    [Fact]
    public void DamagedOrTruncatedEntriesAreIgnoredAndRemoved()
    {
        var cache = new FileCompilationCache(_directory);
        cache.Put("kernel/a", Enumerable.Range(0, 100).Select(i => (byte)i).ToArray());
        var file = Assert.Single(Directory.GetFiles(_directory));

        var bytes = File.ReadAllBytes(file);
        bytes[^1] ^= 0xFF;
        File.WriteAllBytes(file, bytes);
        Assert.False(cache.TryGet("kernel/a", out _));
        Assert.Equal(1, cache.Rejected);
        Assert.False(File.Exists(file));

        cache.Put("kernel/a", new byte[] { 7, 7, 7, 7 });
        File.WriteAllBytes(file, File.ReadAllBytes(file)[..^2]);
        Assert.False(cache.TryGet("kernel/a", out _));
        Assert.Equal(2, cache.Rejected);
    }

    [Fact]
    public void AnEntryWrittenForAnotherKeyIsNotServed()
    {
        var cache = new FileCompilationCache(_directory);
        cache.Put("kernel/a", new byte[] { 1 });
        var fileOfA = Assert.Single(Directory.GetFiles(_directory));
        cache.Put("kernel/b", new byte[] { 2 });
        var fileOfB = Directory.GetFiles(_directory).Single(f => f != fileOfA);
        // As if the hash of "kernel/b" collided with the file of "kernel/a".
        File.Copy(fileOfA, fileOfB, overwrite: true);
        Assert.False(cache.TryGet("kernel/b", out _));
        Assert.True(cache.TryGet("kernel/a", out _));
    }

    [Fact]
    public void TheLeastRecentlyUsedEntriesAreEvictedPastTheCap()
    {
        var cache = new FileCompilationCache(_directory, maxBytes: 10_000);
        for (int i = 0; i < 30; i++)
        {
            cache.Put($"kernel/{i}", new byte[1000]);
            File.SetLastWriteTimeUtc(Directory.GetFiles(_directory).OrderByDescending(File.GetLastWriteTimeUtc).First(), DateTime.UtcNow.AddMinutes(i - 100));
        }

        cache.Trim();
        long total = Directory.GetFiles(_directory).Sum(f => new FileInfo(f).Length);
        Assert.True(total <= 10_000, $"{total} bytes left");
        Assert.True(cache.TryGet("kernel/29", out _));
        Assert.False(cache.TryGet("kernel/0", out _));
    }

    [Fact]
    public async Task ConcurrentWritersLeaveOnlyCompleteEntries()
    {
        var caches = Enumerable.Range(0, 4).Select(_ => new FileCompilationCache(_directory)).ToArray();
        await Task.WhenAll(caches.Select((cache, writer) => Task.Run(() =>
        {
            for (int i = 0; i < 50; i++)
            {
                cache.Put($"kernel/{i % 10}", Enumerable.Repeat((byte)writer, 4096).ToArray());
            }
        }, TestContext.Current.CancellationToken)));

        var reader = new FileCompilationCache(_directory);
        for (int i = 0; i < 10; i++)
        {
            Assert.True(reader.TryGet($"kernel/{i}", out var data));
            Assert.Equal(4096, data.Length);
            Assert.All(data.ToArray(), b => Assert.Equal(data.Span[0], b));
        }

        Assert.Empty(Directory.GetFiles(_directory, "*.tmp"));
    }

    [Fact]
    public void TheEnvironmentRelocatesOrDisablesTheCache()
    {
        var previous = Environment.GetEnvironmentVariable(FileCompilationCache.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(FileCompilationCache.EnvironmentVariable, null);
            Assert.Equal("/default", FileCompilationCache.ResolveDirectory("/default"));
            Environment.SetEnvironmentVariable(FileCompilationCache.EnvironmentVariable, "off");
            Assert.Null(FileCompilationCache.ResolveDirectory("/default"));
            Environment.SetEnvironmentVariable(FileCompilationCache.EnvironmentVariable, _directory);
            Assert.Equal(_directory, FileCompilationCache.ResolveDirectory("/default"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(FileCompilationCache.EnvironmentVariable, previous);
        }
    }
}
