using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// The trellis types of agentionai's llama.cpp fork (ggml ids 144-149, the Gyro TQ*_0 releases): their ids map to the neutral trellis
/// data types and the managed decoder gives the values of a transcription of the fork's decoders (the fixture of
/// <c>Sharplight.Cpu.Tests</c>, <c>tools/gen-trellis-reference.py</c>).
/// </summary>
public class TrellisTypeTests
{
    [Fact]
    public void ForkIds_MapToTheTrellisTypes()
    {
        DataType[] expected =
        [
            DataType.TrellisV4K8_G128, DataType.TrellisV4K6_G128, DataType.TrellisV4K7_G128,
            DataType.TrellisV2K5_G128, DataType.TrellisV2K6_G128, DataType.TrellisV2K7_G128,
        ];
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], GgmlTypes.TryGetDataType(144 + i));
            Assert.Equal(144 + i, GgmlTypes.GetGgmlType(expected[i]));
        }

        Assert.Null(GgmlTypes.TryGetDataType(150));
    }

    [Fact]
    public void ManagedDecoder_MatchesTheReference()
    {
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "trellis_ref.bin")));
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            var type = GgmlTypes.GetDataType(reader.ReadInt32());
            long elements = (long)reader.ReadInt32() * reader.ReadInt32();
            var blocks = reader.ReadBytes((int)type.ByteSize(elements));
            var expected = new float[elements];
            for (int j = 0; j < expected.Length; j++)
            {
                expected[j] = reader.ReadSingle();
            }

            Assert.Equal(expected, GgmlBlockDecoder.Decode(type, blocks));
        }
    }
}
