using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Tests;

/// <summary>
/// Weights written into one temporary file, as a model file stores them (each at an offset that is not a multiple of the page size), and
/// read back through <see cref="FileRangeSource"/>: what routed experts left on disk read. Deletes the file when disposed.
/// </summary>
internal sealed class FileWeights : IDisposable
{
    private readonly FileStream _file;

    public FileWeights()
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"sharplight-weights-{Guid.NewGuid():N}.bin");
        _file = new FileStream(Path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        // A header, so that no weight starts on a page boundary.
        _file.Write(new byte[96]);
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>Appends <paramref name="data"/> as an f32 weight of shape <paramref name="dims"/>.</summary>
    public WeightInfo AddF32(string name, float[] data, params long[] dims) => Add(name, DataType.F32, MemoryMarshal.AsBytes(data.AsSpan()), dims);

    /// <summary>Appends raw bytes as a weight of type <paramref name="type"/> and shape <paramref name="dims"/>.</summary>
    public WeightInfo Add(string name, DataType type, ReadOnlySpan<byte> bytes, params long[] dims)
    {
        long offset = _file.Length;
        _file.Seek(0, SeekOrigin.End);
        _file.Write(bytes);
        // Pad to 32 bytes, as GGUF aligns its tensors.
        _file.Write(new byte[(32 - bytes.Length % 32) % 32]);
        _file.Flush();
        return new WeightInfo { Name = name, Type = type, Shape = new Shape([.. dims.Select(d => (Dim)d)]), Source = new FileRangeSource(Path, offset, bytes.Length) };
    }

    /// <summary>
    /// Appends <paramref name="bytes"/> (a weight of type <paramref name="type"/>) stored in <paramref name="encoding"/>'s form: the file holds
    /// the encoded blocks, the source decodes them (<see cref="EncodedFileRangeSource"/>).
    /// </summary>
    public WeightInfo AddEncoded(string name, DataType type, ReadOnlySpan<byte> bytes, ReversedBlockEncoding encoding, params long[] dims)
    {
        var stored = ReversedBlockEncoding.Encode(bytes, encoding.BlockBytes);
        long offset = _file.Length;
        _file.Seek(0, SeekOrigin.End);
        _file.Write(stored);
        _file.Write(new byte[(32 - stored.Length % 32) % 32]);
        _file.Flush();
        return new WeightInfo
        {
            Name = name, Type = type, Shape = new Shape([.. dims.Select(d => (Dim)d)]), Source = new EncodedFileRangeSource(Path, offset, stored.Length, encoding),
        };
    }

    public void Dispose()
    {
        _file.Dispose();
        File.Delete(Path);
    }
}
