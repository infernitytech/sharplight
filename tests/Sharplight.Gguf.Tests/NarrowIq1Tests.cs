using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// The narrow IQ1 types of unsloth's llama.cpp fork (ggml ids 64-66), widened to IQ1_S at load, against the fork's own decoder:
/// the fixture (<c>tools/gen-narrow-iq1-reference.py</c>) holds blocks quantized from gaussian data and random bytes with the floats
/// the fork decoded them to.
/// </summary>
public class NarrowIq1Tests
{
    [Fact]
    public void WidenedBlocks_DecodeLikeTheFork()
    {
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_narrow_iq1.bin")));
        int count = reader.ReadInt32();
        var seen = new HashSet<int>();
        for (int i = 0; i < count; i++)
        {
            int ggmlType = reader.ReadInt32();
            long elements = (long)reader.ReadInt32() * reader.ReadInt32();
            Assert.Equal(DataType.IQ1_S, GgmlTypes.TryGetDataType(ggmlType));
            int blocks = (int)(elements / 256);
            int stored = GgmlNarrowIq1.StoredBlockBytes(ggmlType);
            var narrow = reader.ReadBytes(blocks * stored);
            var expected = new float[elements];
            for (int j = 0; j < expected.Length; j++)
            {
                expected[j] = reader.ReadSingle();
            }

            var wide = new byte[blocks * GgmlNarrowIq1.WideBlockBytes];
            for (int b = 0; b < blocks; b++)
            {
                GgmlNarrowIq1.Widen(ggmlType, narrow.AsSpan(b * stored, stored), wide.AsSpan(b * GgmlNarrowIq1.WideBlockBytes));
            }

            var actual = GgmlBlockDecoder.Decode(DataType.IQ1_S, wide);
            for (int j = 0; j < expected.Length; j++)
            {
                float e = expected[j];
                float a = actual[j];
                bool same = e.Equals(a) || (float.IsNaN(e) && float.IsNaN(a)) || Math.Abs(e - a) <= 1e-6f * Math.Max(1f, Math.Abs(e));
                Assert.True(same, $"type {ggmlType} record {i} element {j}: expected {e}, got {a}");
            }

            seen.Add(ggmlType);
        }

        Assert.Equal([64, 65, 66], seen.Order());
    }

    [Fact]
    public void Source_ReadsWidenedBlocksAtAnyOffset()
    {
        var narrow = new byte[3 * GgmlNarrowIq1.StoredBlockBytes(GgmlNarrowIq1.Iq1Xs)];
        new Random(7).NextBytes(narrow);
        var expected = new byte[3 * GgmlNarrowIq1.WideBlockBytes];
        for (int b = 0; b < 3; b++)
        {
            GgmlNarrowIq1.Widen(GgmlNarrowIq1.Iq1Xs, narrow.AsSpan(b * 46, 46), expected.AsSpan(b * GgmlNarrowIq1.WideBlockBytes));
        }

        string path = Path.Combine(Path.GetTempPath(), $"narrow-iq1-{Guid.NewGuid():N}.gguf");
        try
        {
            WriteSingleTensorFile(path, GgmlNarrowIq1.Iq1Xs, 768, narrow);
            using var file = GgufFile.Open(path);
            var weight = file.Tensors[0].ToWeightInfo();
            Assert.Equal(DataType.IQ1_S, weight.Type);
            Assert.Equal(expected.Length, weight.Source.ByteLength);
            var all = new byte[expected.Length];
            weight.Source.Read(0, all);
            Assert.Equal(expected, all);
            var part = new byte[61];
            weight.Source.Read(37, part);
            Assert.Equal(expected.AsSpan(37, 61).ToArray(), part);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void WriteSingleTensorFile(string path, int ggmlType, long length, byte[] data)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write("GGUF"u8);
        w.Write(3u);
        w.Write(1UL);
        w.Write(0UL);
        var name = "w"u8.ToArray();
        w.Write((ulong)name.Length);
        w.Write(name);
        w.Write(1u);
        w.Write((ulong)length);
        w.Write((uint)ggmlType);
        w.Write(0UL);
        while (w.BaseStream.Position % 32 != 0)
        {
            w.Write((byte)0);
        }

        w.Write(data);
    }
}
