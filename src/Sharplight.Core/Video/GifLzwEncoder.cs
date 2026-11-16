namespace Sharplight.Video;

/// <summary>Variable-width LZW compression of GIF image data (codes up to 12 bits, clear code when the table is full).</summary>
internal sealed class GifLzwEncoder
{
    private const int MaxCodes = 4096;

    private readonly int _minimumCodeSize;
    private readonly int _clearCode;
    private readonly int _endCode;
    private readonly int[] _hashKeys = new int[MaxCodes * 2 + 1];
    private readonly int[] _hashValues = new int[MaxCodes * 2 + 1];
    private readonly List<byte> _bytes = new(4096);
    private int _bitBuffer;
    private int _bitCount;

    public GifLzwEncoder(int bitsPerPixel)
    {
        _minimumCodeSize = Math.Max(2, bitsPerPixel);
        _clearCode = 1 << _minimumCodeSize;
        _endCode = _clearCode + 1;
    }

    /// <summary>The minimum code size byte that precedes the data blocks.</summary>
    public int MinimumCodeSize => _minimumCodeSize;

    /// <summary>Compresses palette indices and returns the code stream (not yet split into sub-blocks).</summary>
    public byte[] Encode(ReadOnlySpan<byte> indices)
    {
        _bytes.Clear();
        _bitBuffer = 0;
        _bitCount = 0;
        int codeSize = _minimumCodeSize + 1;
        int nextCode = _endCode + 1;
        Array.Fill(_hashKeys, -1);
        Emit(_clearCode, codeSize);
        int prefix = indices[0];
        for (int i = 1; i < indices.Length; i++)
        {
            int suffix = indices[i];
            int key = (prefix << 8) | suffix;
            int slot = Find(key);
            if (_hashKeys[slot] == key)
            {
                prefix = _hashValues[slot];
                continue;
            }

            Emit(prefix, codeSize);
            if (nextCode < MaxCodes)
            {
                _hashKeys[slot] = key;
                _hashValues[slot] = nextCode;
                if (nextCode == (1 << codeSize))
                {
                    codeSize++;
                }

                nextCode++;
            }
            else
            {
                Emit(_clearCode, codeSize);
                Array.Fill(_hashKeys, -1);
                codeSize = _minimumCodeSize + 1;
                nextCode = _endCode + 1;
            }

            prefix = suffix;
        }

        Emit(prefix, codeSize);
        Emit(_endCode, codeSize);
        if (_bitCount > 0)
        {
            _bytes.Add((byte)_bitBuffer);
        }

        return [.. _bytes];
    }

    private int Find(int key)
    {
        int slot = (int)((uint)key * 2654435761u % (uint)_hashKeys.Length);
        while (_hashKeys[slot] != -1 && _hashKeys[slot] != key)
        {
            slot = slot + 1 == _hashKeys.Length ? 0 : slot + 1;
        }

        return slot;
    }

    private void Emit(int code, int codeSize)
    {
        _bitBuffer |= code << _bitCount;
        _bitCount += codeSize;
        while (_bitCount >= 8)
        {
            _bytes.Add((byte)_bitBuffer);
            _bitBuffer >>= 8;
            _bitCount -= 8;
        }
    }
}
