using System.Runtime.InteropServices;
using Sharplight.Runtime.Experts;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>The expert readers (io_uring where the kernel allows it, positional reads on threads) read the same bytes as a plain file read.</summary>
public sealed unsafe class ExpertReaderTests
{
    private const int FileBytes = (3 << 20) + 12345;

    [Fact]
    public void ThreadsReadTheFileBytes() => AssertReadsMatch(ExpertReaders.CreateThreads(4));

    [Fact]
    public void IoUringReadsTheFileBytes()
    {
        var reader = ExpertReaders.TryCreateIoUring(16, out string? reason);
        if (reader is null)
        {
            Assert.Skip($"io_uring is unavailable here: {reason}");
        }

        AssertReadsMatch(reader);
    }

    /// <summary>The default reader is io_uring where it works and the threads otherwise, and says why it fell back.</summary>
    [Fact]
    public void TheDefaultReaderFallsBackToThreads()
    {
        var messages = new List<string>();
        using var reader = ExpertReaders.CreateDefault(messages.Add);
        bool ring = ExpertReaders.TryCreateIoUring(4, out _) is { } probe && Dispose(probe);
        Assert.StartsWith(ring ? "io_uring" : "threads", reader.Name, StringComparison.Ordinal);
        Assert.Equal(!ring && OperatingSystem.IsLinux(), messages.Count == 1);
    }

    private static bool Dispose(IExpertReader reader)
    {
        reader.Dispose();
        return true;
    }

    private static void AssertReadsMatch(IExpertReader reader)
    {
        using var _ = reader;
        string path = Path.Combine(Path.GetTempPath(), $"sharplight-reader-{Guid.NewGuid():N}.bin");
        var data = new byte[FileBytes];
        new Random(7).NextBytes(data);
        File.WriteAllBytes(path, data);
        try
        {
            using var file = reader.Open(path);
            Assert.Equal(FileBytes, file.Length);
            long alignment = Math.Max(1, file.Alignment);
            var random = new Random(3);
            // Aligned ranges anywhere in the file, the last one running past its end (it returns the bytes up to the end).
            var ranges = Enumerable.Range(0, 24)
                .Select(i => (Offset: random.Next(0, FileBytes / (int)alignment) * alignment, Length: (random.Next(1, 64) * 16384L + alignment - 1) / alignment * alignment))
                .Append((Offset: (FileBytes - 5000) / alignment * alignment, Length: 3 * alignment + 8192))
                .ToList();
            long total = ranges.Sum(r => r.Length);
            byte* buffer = (byte*)NativeMemory.AlignedAlloc((nuint)total, 4096);
            try
            {
                var reads = new List<ExpertRead>();
                long at = 0;
                foreach (var (offset, length) in ranges)
                {
                    reads.Add(new ExpertRead(file, offset, (int)length, ExpertReadDestination.Host((nint)(buffer + at))));
                    at += length;
                }

                var batch = reader.Submit(CollectionsMarshal.AsSpan(reads));
                batch.Wait();
                Assert.True(batch.IsCompleted);
                Assert.Equal(ranges.Sum(r => Math.Min(r.Length, FileBytes - r.Offset)), batch.BytesRead);
                at = 0;
                foreach (var (offset, length) in ranges)
                {
                    int valid = (int)Math.Min(length, FileBytes - offset);
                    Assert.True(new ReadOnlySpan<byte>(buffer + at, valid).SequenceEqual(data.AsSpan((int)offset, valid)), $"range at {offset}");
                    at += length;
                }
            }
            finally
            {
                NativeMemory.AlignedFree(buffer);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
