using System.Buffers.Binary;

namespace Sharplight.Vulkan.Tests;

/// <summary>Stored device code is checked completely before a byte reaches the driver (no GPU needed).</summary>
public sealed class DeviceCodeValidationTests
{
    private static readonly byte[] s_pipelineKey = [.. Enumerable.Range(1, 20).Select(i => (byte)i)];

    [Fact]
    public void KernelBinaryEntriesRoundTrip()
    {
        var entry = new PipelineBinaryEntry(s_pipelineKey, [([9, 8, 7], [1, 2, 3, 4, 5]), ([6], [42])]);
        Assert.True(PipelineBinaryCodec.TryDecode(PipelineBinaryCodec.Encode(entry), out var decoded));
        Assert.Equal(entry.PipelineKey, decoded!.PipelineKey);
        Assert.Equal(2, decoded.Binaries.Count);
        Assert.Equal(entry.Binaries[0].Key, decoded.Binaries[0].Key);
        Assert.Equal(entry.Binaries[0].Data, decoded.Binaries[0].Data);
        Assert.Equal(entry.Binaries[1].Data, decoded.Binaries[1].Data);
    }

    [Fact]
    public void DamagedTruncatedOrForeignKernelBinaryEntriesAreRejected()
    {
        var bytes = PipelineBinaryCodec.Encode(new PipelineBinaryEntry(s_pipelineKey, [([9, 8, 7], [1, 2, 3, 4, 5])]));
        for (int cut = 0; cut < bytes.Length; cut++)
        {
            Assert.False(PipelineBinaryCodec.TryDecode(bytes.AsSpan(0, cut), out _), $"accepted a prefix of {cut} bytes");
        }

        var flipped = bytes.ToArray();
        flipped[^2] ^= 1;
        Assert.False(PipelineBinaryCodec.TryDecode(flipped, out _));

        var trailing = bytes.Append((byte)0).ToArray();
        Assert.False(PipelineBinaryCodec.TryDecode(trailing, out _));

        var otherVersion = bytes.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(otherVersion.AsSpan(4), PipelineBinaryCodec.Version + 1);
        Assert.False(PipelineBinaryCodec.TryDecode(otherVersion, out _));

        var otherMagic = bytes.ToArray();
        otherMagic[0] = (byte)'X';
        Assert.False(PipelineBinaryCodec.TryDecode(otherMagic, out _));
    }

    [Fact]
    public void PipelineCacheBlobsMustNameTheDeviceAndDriverBuild()
    {
        const uint vendor = 0x10de;
        const uint device = 0x2bb1;
        var uuid = Enumerable.Range(0, 16).Select(i => (byte)(0xA0 + i)).ToArray();
        var blob = new byte[PipelineCacheHeader.Size + 100];
        BinaryPrimitives.WriteUInt32LittleEndian(blob, PipelineCacheHeader.Size);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(4), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(8), vendor);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(12), device);
        uuid.CopyTo(blob, 16);
        var hex = Convert.ToHexString(uuid);

        Assert.True(PipelineCacheHeader.Matches(blob, vendor, device, hex));
        Assert.False(PipelineCacheHeader.Matches(blob, vendor, device + 1, hex));
        Assert.False(PipelineCacheHeader.Matches(blob, vendor + 1, device, hex));
        Assert.False(PipelineCacheHeader.Matches(blob, vendor, device, Convert.ToHexString(new byte[16])));
        Assert.False(PipelineCacheHeader.Matches(blob.AsSpan(0, 31), vendor, device, hex));
        var badVersion = blob.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(badVersion.AsSpan(4), 2);
        Assert.False(PipelineCacheHeader.Matches(badVersion, vendor, device, hex));
    }
}
