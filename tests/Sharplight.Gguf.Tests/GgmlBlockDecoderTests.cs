using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// The importer's managed block decoder (used for layout changes at import, e.g. splitting a latent-attention <c>kv_b</c> of any
/// quantization) against ggml: the fixture of <c>Sharplight.Cpu.Tests</c> (<c>tools/gen-quant-reference.py</c>) holds blocks of every
/// ggml format, quantized from gaussian data and random bytes, with the floats ggml's own decoder produced.
/// </summary>
public class GgmlBlockDecoderTests
{
    private static readonly Dictionary<int, DataType> s_types = new()
    {
        [2] = DataType.Q4_0, [3] = DataType.Q4_1, [6] = DataType.Q5_0, [7] = DataType.Q5_1, [8] = DataType.Q8_0,
        [10] = DataType.Q2_K, [11] = DataType.Q3_K, [12] = DataType.Q4_K, [13] = DataType.Q5_K, [14] = DataType.Q6_K,
        [16] = DataType.IQ2_XXS, [17] = DataType.IQ2_XS, [18] = DataType.IQ3_XXS, [19] = DataType.IQ1_S, [20] = DataType.IQ4_NL,
        [21] = DataType.IQ3_S, [22] = DataType.IQ2_S, [23] = DataType.IQ4_XS, [29] = DataType.IQ1_M, [34] = DataType.TQ1_0,
        [35] = DataType.TQ2_0, [39] = DataType.MXFP4, [40] = DataType.NVFP4,
    };

    [Fact]
    public void EveryFormat_DecodesLikeGgml()
    {
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_quant_ext.bin")));
        int count = reader.ReadInt32();
        var seen = new HashSet<DataType>();
        for (int i = 0; i < count; i++)
        {
            var type = s_types[reader.ReadInt32()];
            long elements = (long)reader.ReadInt32() * reader.ReadInt32();
            var blocks = reader.ReadBytes((int)type.ByteSize(elements));
            var expected = new float[elements];
            for (int j = 0; j < expected.Length; j++)
            {
                expected[j] = reader.ReadSingle();
            }

            var actual = GgmlBlockDecoder.Decode(type, blocks);
            Assert.Equal(expected.Length, actual.Length);
            for (int j = 0; j < expected.Length; j++)
            {
                // ggml's float decoders and the managed ones may differ in the last bit where ggml fuses multiply and add.
                float e = expected[j];
                float a = actual[j];
                bool same = e.Equals(a) || (float.IsNaN(e) && float.IsNaN(a)) || Math.Abs(e - a) <= 1e-6f * Math.Max(1f, Math.Abs(e));
                Assert.True(same,
                    $"{type.Name} record {i} element {j}: expected {e}, got {a}");
            }

            seen.Add(type);
        }

        Assert.Equal(s_types.Count, seen.Count);
    }
}
