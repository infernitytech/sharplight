using System.Buffers.Binary;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.Gguf.Tests;

/// <summary>
/// I2_S, the ternary type of Microsoft's BitNet fork of llama.cpp (ggml id 36): the codes of a tensor followed by one f32 scale,
/// read as <see cref="DataType.Q2S_G128"/> blocks. The real-file test checks every tensor of a layer of
/// bitnet-b1.58-2B-4T-gguf against tests/Sharplight.Families.Tests/tools/gguf_i2s_reference.py (skipped when the file or its
/// reference is absent).
/// </summary>
public class I2sTests
{
    private static readonly string s_directory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf", "bitnet-2b-i2s");

    [Fact]
    public void Source_SpreadsTheTensorScaleOverTheBlocks()
    {
        const int rows = 3, columns = 256;
        var codes = new byte[rows * columns / 4];
        new Random(5).NextBytes(codes);
        for (int i = 0; i < codes.Length; i++)
        {
            // Codes 0..2 only: the byte's four fields.
            codes[i] = (byte)((codes[i] % 3) << 6 | (codes[i] / 3 % 3) << 4 | (codes[i] / 9 % 3) << 2 | (codes[i] / 27 % 3));
        }

        var file = new byte[codes.Length + 32];
        codes.CopyTo(file, 0);
        BinaryPrimitives.WriteSingleLittleEndian(file.AsSpan(codes.Length), 1.7f);
        string path = Path.Combine(Path.GetTempPath(), $"i2s-{Guid.NewGuid():N}.gguf");
        try
        {
            WriteTwoDimensionalFile(path, columns, rows, file);
            using var gguf = GgufFile.Open(path);
            var weight = gguf.Tensors[0].ToWeightInfo();
            Assert.Equal(DataType.Q2S_G128, weight.Type);
            Assert.Equal(rows * columns / 128 * 36, weight.Source.ByteLength);
            var blocks = new byte[weight.Source.ByteLength];
            weight.Source.Read(0, blocks);
            var part = new byte[50];
            weight.Source.Read(41, part);
            Assert.Equal(blocks.AsSpan(41, 50).ToArray(), part);
            var decoded = GgmlBlockDecoder.Decode(DataType.Q2S_G128, blocks);
            for (int i = 0; i < decoded.Length; i++)
            {
                int group = i / 128, within = i % 128;
                int code = (codes[group * 32 + within % 32] >> (6 - 2 * (within / 32))) & 3;
                Assert.Equal((code - 1) * 1.7f, decoded[i]);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RealFile_DequantizesLikeTheReference()
    {
        string path = Path.Combine(s_directory, "ggml-model-i2_s.gguf");
        string reference = Path.Combine(s_directory, "sharplight-reference");
        Assert.SkipUnless(File.Exists(path) && Directory.Exists(reference), $"{path} or its reference not present");
        using var gguf = GgufFile.Open(path);
        int checkedTensors = 0;
        foreach (string file in Directory.GetFiles(reference, "*.f32"))
        {
            string tensor = Path.GetFileNameWithoutExtension(file);
            var weight = gguf.GetTensor(tensor).ToWeightInfo();
            Assert.Equal(DataType.Q2S_G128, weight.Type);
            var bytes = new byte[weight.Source.ByteLength];
            weight.Source.Read(0, bytes);
            var actual = GgmlBlockDecoder.Decode(DataType.Q2S_G128, bytes);
            var library = new float[actual.Length];
            System.Buffer.BlockCopy(File.ReadAllBytes(file), 0, library, 0, library.Length * 4);
            Assert.Equal(library.Length, actual.Length);
            int mismatches = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(library[i]) && !(actual[i] == 0 && library[i] == 0))
                {
                    mismatches++;
                }
            }

            Assert.True(mismatches == 0, $"{tensor}: {mismatches} values differ from the reference");
            checkedTensors++;
        }

        Assert.True(checkedTensors > 0);
    }

    private static void WriteTwoDimensionalFile(string path, long columns, long rows, byte[] data)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write("GGUF"u8);
        w.Write(3u);
        w.Write(1UL);
        w.Write(0UL);
        var name = "w"u8.ToArray();
        w.Write((ulong)name.Length);
        w.Write(name);
        w.Write(2u);
        w.Write((ulong)columns);
        w.Write((ulong)rows);
        w.Write((uint)GgmlTypes.I2S);
        w.Write(0UL);
        while (w.BaseStream.Position % 32 != 0)
        {
            w.Write((byte)0);
        }

        w.Write(data);
    }
}
