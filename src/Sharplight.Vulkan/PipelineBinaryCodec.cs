using System.Buffers.Binary;
using System.Security.Cryptography;

namespace Sharplight.Vulkan;

/// <summary>
/// Serializes <see cref="PipelineBinaryEntry"/> values (artifact entries <c>vulkan/pipeline_binary/&lt;kernel key&gt;</c>):
/// <c>"SLPB"</c>, <c>u32 version</c>, the pipeline key, <c>u32 count</c>, then per binary its key, <c>u64 size</c>, the SHA-256 of
/// the data and the data (little endian; keys as <c>u32 size</c> plus <see cref="MaxKeySize"/> bytes). Every entry is checked
/// completely before any byte reaches the driver: a truncated, corrupted or foreign entry is rejected, never trusted.
/// </summary>
internal static class PipelineBinaryCodec
{
    /// <summary>VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR.</summary>
    public const int MaxKeySize = 32;

    /// <summary>Format version written.</summary>
    public const int Version = 1;

    private const int HashSize = 32;
    private const int KeyFieldSize = sizeof(uint) + MaxKeySize;
    private static ReadOnlySpan<byte> Magic => "SLPB"u8;

    /// <summary>Encodes an entry.</summary>
    /// <exception cref="ArgumentException">A key is longer than <see cref="MaxKeySize"/>.</exception>
    public static byte[] Encode(PipelineBinaryEntry entry)
    {
        long size = Magic.Length + sizeof(uint) + KeyFieldSize + sizeof(uint)
            + entry.Binaries.Sum(b => (long)KeyFieldSize + sizeof(ulong) + HashSize + b.Data.Length);
        var buffer = new byte[size];
        var span = buffer.AsSpan();
        Magic.CopyTo(span);
        int position = Magic.Length;
        BinaryPrimitives.WriteUInt32LittleEndian(span[position..], Version);
        position += sizeof(uint);
        position += WriteKey(span[position..], entry.PipelineKey);
        BinaryPrimitives.WriteUInt32LittleEndian(span[position..], (uint)entry.Binaries.Count);
        position += sizeof(uint);
        foreach (var (key, data) in entry.Binaries)
        {
            position += WriteKey(span[position..], key);
            BinaryPrimitives.WriteUInt64LittleEndian(span[position..], (ulong)data.Length);
            position += sizeof(ulong);
            SHA256.HashData(data, span.Slice(position, HashSize));
            position += HashSize;
            data.CopyTo(span[position..]);
            position += data.Length;
        }

        return buffer;
    }

    /// <summary>Decodes and verifies an entry.</summary>
    /// <param name="data">The stored bytes.</param>
    /// <param name="entry">The entry when valid.</param>
    /// <returns>False for anything but a complete, intact entry of this format version.</returns>
    public static bool TryDecode(ReadOnlySpan<byte> data, out PipelineBinaryEntry? entry)
    {
        entry = null;
        int position = 0;
        if (data.Length < Magic.Length + sizeof(uint) || !data[..Magic.Length].SequenceEqual(Magic))
        {
            return false;
        }

        position += Magic.Length;
        if (BinaryPrimitives.ReadUInt32LittleEndian(data[position..]) != Version)
        {
            return false;
        }

        position += sizeof(uint);
        if (!TryReadKey(data, ref position, out var pipelineKey) || !TryReadUInt32(data, ref position, out uint count) || count == 0 || count > 64)
        {
            return false;
        }

        var binaries = new List<(byte[], byte[])>((int)count);
        Span<byte> actual = stackalloc byte[HashSize];
        for (int i = 0; i < count; i++)
        {
            if (!TryReadKey(data, ref position, out var key) || data.Length - position < sizeof(ulong) + HashSize)
            {
                return false;
            }

            ulong size = BinaryPrimitives.ReadUInt64LittleEndian(data[position..]);
            position += sizeof(ulong);
            var hash = data.Slice(position, HashSize);
            position += HashSize;
            if (size == 0 || size > (ulong)(data.Length - position))
            {
                return false;
            }

            var bytes = data.Slice(position, (int)size);
            position += (int)size;
            SHA256.HashData(bytes, actual);
            if (!actual.SequenceEqual(hash))
            {
                return false;
            }

            binaries.Add((key, bytes.ToArray()));
        }

        if (position != data.Length)
        {
            return false;
        }

        entry = new PipelineBinaryEntry(pipelineKey, binaries);
        return true;
    }

    private static int WriteKey(Span<byte> destination, byte[] key)
    {
        if (key.Length > MaxKeySize)
        {
            throw new ArgumentException($"Pipeline binary keys have at most {MaxKeySize} bytes.", nameof(key));
        }

        BinaryPrimitives.WriteUInt32LittleEndian(destination, (uint)key.Length);
        key.CopyTo(destination[sizeof(uint)..]);
        destination.Slice(sizeof(uint) + key.Length, MaxKeySize - key.Length).Clear();
        return KeyFieldSize;
    }

    private static bool TryReadKey(ReadOnlySpan<byte> data, ref int position, out byte[] key)
    {
        key = [];
        if (data.Length - position < KeyFieldSize)
        {
            return false;
        }

        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[position..]);
        if (size == 0 || size > MaxKeySize)
        {
            return false;
        }

        key = data.Slice(position + sizeof(uint), (int)size).ToArray();
        position += KeyFieldSize;
        return true;
    }

    private static bool TryReadUInt32(ReadOnlySpan<byte> data, ref int position, out uint value)
    {
        value = 0;
        if (data.Length - position < sizeof(uint))
        {
            return false;
        }

        value = BinaryPrimitives.ReadUInt32LittleEndian(data[position..]);
        position += sizeof(uint);
        return true;
    }
}
