using Sharplight.Models;

namespace Sharplight.Tests;

/// <summary>
/// A weight whose file range holds its blocks in another encoding (<see cref="IStoredBlockEncoding"/>), as a GGUF of a narrow quantization
/// type stores them: <see cref="Read"/> decodes, <see cref="TryGetEncodedFileRange"/> reports the stored range, and there is no plain range.
/// </summary>
internal sealed class EncodedFileRangeSource(string path, long offset, long storedLength, IStoredBlockEncoding encoding) : IWeightSource
{
    public long ByteLength => storedLength / encoding.StoredBlockBytes * encoding.BlockBytes;

    public void Read(long position, Span<byte> destination)
    {
        var decoded = new byte[ByteLength];
        var stored = new byte[storedLength];
        using (var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
        {
            RandomAccess.Read(handle, stored, offset);
        }

        encoding.Decode(stored, decoded);
        decoded.AsSpan((int)position, destination.Length).CopyTo(destination);
    }

    public bool TryGetEncodedFileRange(out string filePath, out long fileOffset, out IStoredBlockEncoding? blockEncoding)
    {
        filePath = path;
        fileOffset = offset;
        blockEncoding = encoding;
        return true;
    }
}
