namespace Sharplight.Models;

internal sealed class DerivedWeightSource(long length, Func<byte[]> produce) : IWeightSource
{
    private Lazy<byte[]> _data = Create(length, produce);

    public long ByteLength => length;

    public void Materialize() => _ = _data.Value;

    public void Read(long offset, Span<byte> destination) => _data.Value.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);

    public void Release()
    {
        if (_data.IsValueCreated)
        {
            Interlocked.Exchange(ref _data, Create(length, produce));
        }
    }

    private static Lazy<byte[]> Create(long length, Func<byte[]> produce) => new(() =>
    {
        var d = produce();
        if (d.LongLength != length)
        {
            throw new InvalidDataException($"Derived weight produced {d.LongLength} bytes, expected {length}.");
        }

        return d;
    });
}
