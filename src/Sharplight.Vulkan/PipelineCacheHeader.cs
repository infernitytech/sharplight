using System.Buffers.Binary;

namespace Sharplight.Vulkan;

/// <summary>
/// The version one header every Vulkan pipeline cache blob starts with (header size, header version, vendor id, device id,
/// pipelineCacheUUID). Checked before a stored blob is handed to the driver, so a blob of another device or driver build (or a
/// damaged one) is ignored instead of trusted.
/// </summary>
internal static class PipelineCacheHeader
{
    /// <summary>Bytes of VkPipelineCacheHeaderVersionOne.</summary>
    public const int Size = 32;

    private const uint VersionOne = 1;
    private const int UuidSize = 16;

    /// <summary>Whether <paramref name="blob"/> is a pipeline cache of the device with these ids and cache UUID (hexadecimal).</summary>
    public static bool Matches(ReadOnlySpan<byte> blob, uint vendorId, uint deviceId, string pipelineCacheUuid)
    {
        if (blob.Length < Size)
        {
            return false;
        }

        uint headerSize = BinaryPrimitives.ReadUInt32LittleEndian(blob);
        uint headerVersion = BinaryPrimitives.ReadUInt32LittleEndian(blob[4..]);
        if (headerSize < Size || headerSize > blob.Length || headerVersion != VersionOne)
        {
            return false;
        }

        return BinaryPrimitives.ReadUInt32LittleEndian(blob[8..]) == vendorId
            && BinaryPrimitives.ReadUInt32LittleEndian(blob[12..]) == deviceId
            && Convert.ToHexString(blob.Slice(16, UuidSize)) == pipelineCacheUuid;
    }
}
