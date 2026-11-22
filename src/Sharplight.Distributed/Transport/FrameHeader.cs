using System.Buffers.Binary;

namespace Sharplight.Distributed.Transport;

/// <summary>
/// The 12-byte frame header: payload length (u32), kind (u8), three reserved bytes, request id (u32), little-endian. The payload follows.
/// </summary>
internal static class FrameHeader
{
    /// <summary>Header size in bytes.</summary>
    public const int Size = 12;

    public static void Write(Span<byte> destination, FrameKind kind, uint requestId, int payloadLength)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(destination, (uint)payloadLength);
        destination[4] = (byte)kind;
        destination[5] = 0;
        destination[6] = 0;
        destination[7] = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(destination[8..], requestId);
    }

    /// <exception cref="InvalidDataException">The header is not one of this protocol.</exception>
    public static (FrameKind Kind, uint RequestId, int Length) Read(ReadOnlySpan<byte> header, int maxPayload)
    {
        uint length = BinaryPrimitives.ReadUInt32LittleEndian(header);
        var kind = (FrameKind)header[4];
        if (length > (uint)maxPayload || !Enum.IsDefined(kind) || header[5] != 0 || header[6] != 0 || header[7] != 0)
        {
            throw new InvalidDataException($"Malformed segment frame header (kind {header[4]}, length {length}).");
        }

        return (kind, BinaryPrimitives.ReadUInt32LittleEndian(header[8..]), (int)length);
    }
}
