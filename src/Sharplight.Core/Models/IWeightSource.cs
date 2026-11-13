namespace Sharplight.Models;

/// <summary>Provides the raw bytes of a weight tensor (e.g. a memory-mapped file region).</summary>
public interface IWeightSource
{
    /// <summary>Size of the stored data in bytes.</summary>
    public long ByteLength { get; }

    /// <summary>Copies <paramref name="destination"/>.Length bytes starting at <paramref name="offset"/>.</summary>
    public void Read(long offset, Span<byte> destination);

    /// <summary>
    /// Returns a pointer to the data if it is directly addressable (memory mapped), enabling zero-copy
    /// uploads; otherwise <see cref="IntPtr.Zero"/>.
    /// </summary>
    public nint TryGetPointer() => 0;

    /// <summary>
    /// Computes the data now if this source produces it lazily on first <see cref="Read"/> (default: nothing to do), so that a
    /// consumer can produce several weights in parallel ahead of reading them. Thread-safe; concurrent reads wait for the result.
    /// </summary>
    public void Materialize()
    {
    }
}
