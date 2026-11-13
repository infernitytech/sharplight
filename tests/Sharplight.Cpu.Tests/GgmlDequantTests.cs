using System.Runtime.InteropServices;
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

    /// <summary>The quantized (non-random) records: their dequantized values are well scaled, so a float matmul is a meaningful reference.</summary>
    public static IEnumerable<object[]> QuantizedRecords() => Records().Where(r => (int)r[0] % 2 == 0);

    [Theory, MemberData(nameof(QuantizedRecords))]
    public void MatMulMatchesDequantizedProduct(int index, string label)
    {
        var record = Load()[index];
        foreach (int tokens in new[] { 1, 5 })
        {
            var x = Kit.Rand(tokens * record.Cols, 12);
            var tg = new TestGraph();
            var w = tg.W("w", record.Type, new Shape(record.Rows, record.Cols), record.Blocks);
            var xin = tg.InF("x", TestGraph.T, record.Cols);
            tg.B.Output("y", tg.B.MatMul(xin, w));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(tokens);
            m.Run(p);
            var expected = new float[tokens * record.Rows];
            for (int t = 0; t < tokens; t++)
            {
                for (int r = 0; r < record.Rows; r++)
                {
                    double sum = 0;
                    for (int c = 0; c < record.Cols; c++)
                    {
                        sum += (double)x[t * record.Cols + c] * record.Expected[r * record.Cols + c];
                    }

                    expected[t * record.Rows + r] = (float)sum;
                }
            }

            Kit.AssertClose(expected, Kit.Out(m, "y", p), 2e-3, 1e-4, label + " matmul");
        }
    }

    /// <summary>MatMulId with the four quantized rows read as two experts of two output rows each.</summary>
    [Theory, MemberData(nameof(QuantizedRecords))]
    public void MatMulIdMatchesDequantizedProduct(int index, string label)
    {
        var record = Load()[index];
        const int experts = 2;
        const int outputs = 2;
        const int tokens = 3;
        int[] ids = [1, 0, 0, 1, 1, 0];   // [T, S] with S = 2 slots
        var x = Kit.Rand(tokens * record.Cols, 12);
        var tg = new TestGraph();
        var w = tg.W("w", record.Type, new Shape(experts, outputs, record.Cols), record.Blocks);
        var xin = tg.InF("x", TestGraph.T, record.Cols);
        var idin = tg.In("ids", DataType.I32, TestGraph.T, 2);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [xin, w, idin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, idin, ids);
        var p = Kit.P(tokens);
        m.Run(p);
        var expected = new float[tokens * 2 * outputs];
        for (int t = 0; t < tokens; t++)
        {
            for (int s = 0; s < 2; s++)
            {
                for (int n = 0; n < outputs; n++)
                {
                    double sum = 0;
                    int row = ids[t * 2 + s] * outputs + n;
                    for (int c = 0; c < record.Cols; c++)
                    {
                        sum += (double)x[t * record.Cols + c] * record.Expected[row * record.Cols + c];
                    }

                    expected[(t * 2 + s) * outputs + n] = (float)sum;
                }
            }
        }

        Kit.AssertClose(expected, Kit.Out(m, "y", p), 2e-3, 1e-4, label + " matmulid");
    }

    [Theory, MemberData(nameof(QuantizedRecords))]
    public void GatherMatchesDequantizedRows(int index, string label)
    {
        var record = Load()[index];
        var tg = new TestGraph();
        var table = tg.W("tbl", record.Type, new Shape(record.Rows, record.Cols), record.Blocks);
        var idx = tg.In("idx", DataType.I32, TestGraph.T);
        tg.B.Output("y", tg.B.Gather(table, idx));
        using var m = Kit.Compile(tg.B.Build());
        int[] rows = [3, 0, 2];
        m.SetInput(idx, MemoryMarshal.AsBytes(rows.AsSpan()));
        var p = Kit.P(rows.Length);
        m.Run(p);
        var expected = rows.SelectMany(r => record.Expected.AsSpan(r * record.Cols, record.Cols).ToArray()).ToArray();
        Kit.AssertClose(expected, Kit.Out(m, "y", p), 0, 0, label + " gather");
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
