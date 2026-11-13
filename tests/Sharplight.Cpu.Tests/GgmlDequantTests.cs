using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Dequantization against llama.cpp: Fixtures/ggml_quant_ext.bin holds blocks (quantized by ggml from gaussian data, and random bytes)
/// with the floats ggml's own decoder produced (tools/gen-quant-reference.py). The new formats must match bit for bit.
/// </summary>
public unsafe class GgmlDequantTests
{
    private static readonly Dictionary<int, DataType> s_types = new()
    {
        [2] = DataType.Q4_0, [3] = DataType.Q4_1, [6] = DataType.Q5_0, [7] = DataType.Q5_1, [8] = DataType.Q8_0,
        [10] = DataType.Q2_K, [11] = DataType.Q3_K, [12] = DataType.Q4_K, [13] = DataType.Q5_K, [14] = DataType.Q6_K,
        [16] = DataType.IQ2_XXS, [17] = DataType.IQ2_XS, [18] = DataType.IQ3_XXS, [19] = DataType.IQ1_S, [20] = DataType.IQ4_NL,
        [21] = DataType.IQ3_S, [22] = DataType.IQ2_S, [23] = DataType.IQ4_XS, [29] = DataType.IQ1_M, [34] = DataType.TQ1_0,
        [35] = DataType.TQ2_0, [39] = DataType.MXFP4, [40] = DataType.NVFP4,
    };

    /// <summary>Pre-existing formats whose dequantizers fuse multiply and add (and may differ in the sign of zero); compared with a tight tolerance instead of bit for bit.</summary>
    private static readonly HashSet<int> s_fusedTypes = [2, 3, 6, 7, 8, 12, 13, 14];

    private sealed record Record(DataType Type, int Rows, int Cols, byte[] Blocks, float[] Expected);

    private static List<Record> Load()
    {
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_quant_ext.bin")));
        int count = reader.ReadInt32();
        var records = new List<Record>();
        for (int i = 0; i < count; i++)
        {
            int id = reader.ReadInt32();
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var type = s_types[id];
            var blocks = reader.ReadBytes((int)type.ByteSize((long)rows * cols));
            var expected = new float[rows * cols];
            for (int j = 0; j < expected.Length; j++)
            {
                expected[j] = reader.ReadSingle();
            }

            records.Add(new Record(type, rows, cols, blocks, expected));
        }

        return records;
    }

    public static IEnumerable<object[]> Records()
    {
        var records = Load();
        for (int i = 0; i < records.Count; i++)
        {
            yield return [i, records[i].Type.Name + (i % 2 == 0 ? " (ggml quantized)" : " (random bytes)")];
        }
    }

    [Theory, MemberData(nameof(Records))]
    public void MatchesGgmlDecoder(int index, string label)
    {
        var record = Load()[index];
        var format = Quant.FormatOf(record.Type) ?? throw new InvalidOperationException(record.Type.Name);
        var actual = new float[record.Expected.Length];
        long rowBytes = record.Type.ByteSize(record.Cols);
        fixed (byte* src = record.Blocks)
        fixed (float* dst = actual)
        {
            for (int r = 0; r < record.Rows; r++)
            {
                Quant.Dequantize(format, src + r * rowBytes, record.Cols, dst + (long)r * record.Cols);
            }
        }

        bool fused = s_fusedTypes.Contains(s_types.First(kv => kv.Value == record.Type).Key);
        if (fused)
        {
            Kit.AssertClose(record.Expected, actual, 1e-5, 1e-5, label);
            return;
        }

        for (int i = 0; i < actual.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(record.Expected[i]) == BitConverter.SingleToInt32Bits(actual[i]) || (float.IsNaN(record.Expected[i]) && float.IsNaN(actual[i])),
                $"{label}: element {i}: expected {record.Expected[i]:G9}, actual {actual[i]:G9}");
        }
    }
}
