namespace Sharplight.Models;

/// <summary>
/// An <see cref="IWeightSource"/> over bytes in memory, produced on first use: attachments that an importer or adapter computes while
/// loading (a serialized tokenizer, a computed filter bank). Thread-safe; the producer runs once.
/// </summary>
public sealed class MemoryWeightSource : IWeightSource
{
    private readonly Lazy<byte[]> _data;

    /// <summary>Creates the source over existing bytes (not copied; must not change afterwards).</summary>
    /// <param name="data">The bytes.</param>
    public MemoryWeightSource(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _data = new Lazy<byte[]>(data);
    }

    /// <summary>Creates the source over bytes computed on first access (<see cref="ByteLength"/>, <see cref="Read"/>).</summary>
    /// <param name="produce">Computes the bytes once.</param>
    public MemoryWeightSource(Func<byte[]> produce)
    {
        ArgumentNullException.ThrowIfNull(produce);
        _data = new Lazy<byte[]>(produce, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    /// <inheritdoc/>
    public long ByteLength => _data.Value.LongLength;

    /// <inheritdoc/>
    public void Materialize() => _ = _data.Value;

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination) => _data.Value.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
}
