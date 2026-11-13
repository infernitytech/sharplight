namespace Sharplight.Tokenizers.Bpe;

/// <summary>Minimal growable int buffer passed by reference through the encoder (avoids <see cref="List{T}"/> overhead on the hot path).</summary>
internal struct IntList
{
    private int[] _buffer;
    private int _count;

    public IntList(int capacity)
    {
        _buffer = new int[capacity];
        _count = 0;
    }

    public void Add(int value)
    {
        if (_count == _buffer.Length)
        {
            Array.Resize(ref _buffer, _buffer.Length * 2);
        }

        _buffer[_count++] = value;
    }

    public void AddRange(ReadOnlySpan<int> values)
    {
        if (_count + values.Length > _buffer.Length)
        {
            Array.Resize(ref _buffer, Math.Max(_buffer.Length * 2, _count + values.Length));
        }

        values.CopyTo(_buffer.AsSpan(_count));
        _count += values.Length;
    }

    public readonly int[] ToArray() => _buffer.AsSpan(0, _count).ToArray();
}
