using Sharplight.Models;

namespace Sharplight.Tests;

/// <summary>A weight stored in a range of a file, read with positional reads (not memory mapped), that reports its file range.</summary>
internal sealed class FileRangeSource(string path, long offset, long length) : IWeightSource
{
    public long ByteLength => length;

    public void Read(long position, Span<byte> destination)
    {
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        RandomAccess.Read(handle, destination, offset + position);
    }

    public bool TryGetFileRange(out string filePath, out long fileOffset)
    {
        filePath = path;
        fileOffset = offset;
        return true;
    }
}
