using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The trellis-coded formats against a transcription of the reference decoders: Fixtures/trellis_ref.bin holds random blocks of
/// each format with the values the reference gives them (tools/gen-trellis-reference.py). They must match bit for bit.
/// </summary>
public unsafe class TrellisDequantTests
{
    /// <summary>The fixture's type ids (the ggml ids of the fork that introduced the formats) and their data types.</summary>
    internal static readonly Dictionary<int, DataType> Types = new()
    {
        [144] = DataType.TrellisV4K8_G128, [145] = DataType.TrellisV4K6_G128, [146] = DataType.TrellisV4K7_G128,
        [147] = DataType.TrellisV2K5_G128, [148] = DataType.TrellisV2K6_G128, [149] = DataType.TrellisV2K7_G128,
    };

    [Fact]
    public void EveryFormat_DequantizesLikeTheReference()
    {
        var seen = new HashSet<DataType>();
        foreach (var (type, rows, cols, blocks, expected) in Load())
        {
            var format = Quant.FormatOf(type) ?? throw new InvalidOperationException(type.Name);
            Assert.Equal(type.BlockSize, Quant.BlockSize(format));
            Assert.Equal(type.BlockBytes, Quant.BlockBytes(format));
            var actual = new float[expected.Length];
            long rowBytes = type.ByteSize(cols);
            fixed (byte* src = blocks)
            fixed (float* dst = actual)
            {
                for (int r = 0; r < rows; r++)
                {
                    Quant.Dequantize(format, src + r * rowBytes, cols, dst + (long)r * cols);
                }
            }

            for (int j = 0; j < expected.Length; j++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(expected[j]) == BitConverter.SingleToInt32Bits(actual[j]),
                    $"{type.Name} element {j}: expected {expected[j]}, got {actual[j]}");
            }

            seen.Add(type);
        }

        Assert.Equal(Types.Count, seen.Count);
    }

    private static IEnumerable<(DataType Type, int Rows, int Cols, byte[] Blocks, float[] Expected)> Load()
    {
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "trellis_ref.bin")));
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            var type = Types[reader.ReadInt32()];
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var blocks = reader.ReadBytes((int)type.ByteSize((long)rows * cols));
            var expected = new float[rows * cols];
            for (int j = 0; j < expected.Length; j++)
            {
                expected[j] = reader.ReadSingle();
            }

            yield return (type, rows, cols, blocks, expected);
        }
    }
}
