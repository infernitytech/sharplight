namespace Sharplight.Models;

internal sealed class StackedWeightSource(IWeightSource[] parts) : IWeightSource
{
    private readonly long _each = parts[0].ByteLength;
    public long ByteLength => _each * parts.Length;
    public void Read(long offset, Span<byte> destination)
    {
        while (destination.Length > 0)
        {
            int i = (int)(offset / _each);
            long inPart = offset - i * _each;
            int n = (int)Math.Min(destination.Length, _each - inPart);
            parts[i].Read(inPart, destination[..n]);
            destination = destination[n..];
            offset += n;
        }
    }
}
