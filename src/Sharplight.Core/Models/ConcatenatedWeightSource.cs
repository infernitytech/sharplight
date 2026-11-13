namespace Sharplight.Models;

/// <summary>The bytes of several sources back to back (parts may differ in length).</summary>
internal sealed class ConcatenatedWeightSource : IWeightSource
{
    private readonly IWeightSource[] _parts;
    private readonly long[] _starts;

    public ConcatenatedWeightSource(IWeightSource[] parts)
    {
        _parts = parts;
        _starts = new long[parts.Length + 1];
        for (int i = 0; i < parts.Length; i++)
        {
            _starts[i + 1] = _starts[i] + parts[i].ByteLength;
        }
    }

    public long ByteLength => _starts[^1];

    public void Materialize()
    {
        foreach (var part in _parts)
        {
            part.Materialize();
        }
    }

    public void Read(long offset, Span<byte> destination)
    {
        int i = Array.BinarySearch(_starts, offset);
        i = i >= 0 ? i : ~i - 1;
        while (destination.Length > 0)
        {
            while (_starts[i + 1] <= offset)
            {
                i++;
            }

            long inPart = offset - _starts[i];
            int n = (int)Math.Min(destination.Length, _parts[i].ByteLength - inPart);
            _parts[i].Read(inPart, destination[..n]);
            destination = destination[n..];
            offset += n;
        }
    }
}
