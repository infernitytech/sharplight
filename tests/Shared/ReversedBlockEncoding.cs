using Sharplight.Models;

namespace Sharplight.Tests;

/// <summary>
/// A test block encoding (<see cref="IStoredBlockEncoding"/>): each stored block is a marker byte, the block's bytes in reverse order and
/// another marker byte, so the stored form is larger than the weight's and decodes to it only through <see cref="Decode"/>.
/// </summary>
internal sealed class ReversedBlockEncoding(int blockBytes) : IStoredBlockEncoding
{
    public int StoredBlockBytes => blockBytes + 2;

    public int BlockBytes => blockBytes;

    public static byte[] Encode(ReadOnlySpan<byte> data, int blockBytes)
    {
        int blocks = data.Length / blockBytes;
        var stored = new byte[blocks * (blockBytes + 2)];
        for (int b = 0; b < blocks; b++)
        {
            var target = stored.AsSpan(b * (blockBytes + 2), blockBytes + 2);
            target[0] = 0xA5;
            data.Slice(b * blockBytes, blockBytes).CopyTo(target[1..]);
            target.Slice(1, blockBytes).Reverse();
            target[^1] = 0x5A;
        }

        return stored;
    }

    public void Decode(ReadOnlySpan<byte> stored, Span<byte> blocks)
    {
        int count = stored.Length / StoredBlockBytes;
        if (count * StoredBlockBytes != stored.Length || blocks.Length != count * BlockBytes)
        {
            throw new ArgumentException("partial blocks");
        }

        for (int b = 0; b < count; b++)
        {
            var source = stored.Slice(b * StoredBlockBytes, StoredBlockBytes);
            if (source[0] != 0xA5 || source[^1] != 0x5A)
            {
                throw new InvalidDataException($"block {b} is not a stored block");
            }

            var target = blocks.Slice(b * BlockBytes, BlockBytes);
            source.Slice(1, BlockBytes).CopyTo(target);
            target.Reverse();
        }
    }
}
