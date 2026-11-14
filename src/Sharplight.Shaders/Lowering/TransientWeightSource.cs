using Sharplight.Models;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// A weight computed on the host for upload only (the split-f16 parts of f32 weights): the bytes are produced on first use
/// and dropped once every byte has been read, so a compiled program does not keep a host copy of what it uploaded. A later
/// read (another device, a recompilation) produces them again. Thread-safe.
/// </summary>
internal sealed class TransientWeightSource(long length, Func<byte[]> produce) : IWeightSource
{
    private readonly Lock _lock = new();
    private byte[]? _data;
    private long _read;

    /// <inheritdoc/>
    public long ByteLength => length;

    /// <inheritdoc/>
    public void Materialize()
    {
        lock (_lock)
        {
            _ = Data();
        }
    }

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination)
    {
        lock (_lock)
        {
            Data().AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
            _read += destination.Length;
            if (_read >= length)
            {
                _data = null;
                _read = 0;
            }
        }
    }

    private byte[] Data()
    {
        if (_data is null)
        {
            var d = produce();
            if (d.LongLength != length)
            {
                throw new InvalidDataException($"Transient weight produced {d.LongLength} bytes, expected {length}.");
            }

            _data = d;
        }

        return _data;
    }
}
