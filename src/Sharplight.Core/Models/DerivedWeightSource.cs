namespace Sharplight.Models;

internal sealed class DerivedWeightSource(long length, Func<byte[]> produce) : IWeightSource
{
    private readonly Lazy<byte[]> _data = new(() =>
    {
        var d = produce();
        if (d.LongLength != length)
        {
            throw new InvalidDataException($"Derived weight produced {d.LongLength} bytes, expected {length}.");
        }

        return d;
    });
    public long ByteLength => length;
    public void Materialize() => _ = _data.Value;
    public void Read(long offset, Span<byte> destination) => _data.Value.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
}
