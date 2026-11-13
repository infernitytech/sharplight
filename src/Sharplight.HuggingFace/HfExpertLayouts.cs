using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Lazy conversions of fused expert tensors whose Hugging Face layout differs from the canonical rank-3 [E, out, in] weights: the
/// [E, in, out] layout of gpt-oss and Llama 4, interleaved gate/up columns (gpt-oss), and gpt-oss MXFP4 blocks.
/// </summary>
internal static class HfExpertLayouts
{
    /// <summary>The rows <c>0, 2, 4, …, 1, 3, 5, …</c> of an interleaved [gate, up] pair layout, gate rows first.</summary>
    public static int[] DeinterleavedPairs(int count)
        => [.. Enumerable.Range(0, count / 2).Select(i => 2 * i), .. Enumerable.Range(0, count / 2).Select(i => 2 * i + 1)];

    /// <summary>
    /// The canonical [E, out, in] weight of an expert tensor stored as [E, in, out], optionally keeping only the output rows in
    /// <paramref name="rowOrder"/> (in that order). Dense float storage types only.
    /// </summary>
    public static WeightInfo Transposed(WeightInfo w, string name, int[]? rowOrder = null)
    {
        long experts = w.Shape[0].Fixed;
        long inner = w.Shape[1].Fixed;
        long outer = w.Shape[2].Fixed;
        int[] rows = rowOrder ?? [.. Enumerable.Range(0, checked((int)outer))];
        int elementBytes = checked((int)w.Type.ByteSize(1));
        return WeightInfo.Derived(name, w.Type, new Shape(experts, rows.Length, inner), () =>
        {
            var source = w.ReadAll();
            var result = new byte[experts * rows.Length * inner * elementBytes];
            for (long e = 0; e < experts; e++)
            {
                long sourceBase = e * inner * outer;
                long resultBase = e * rows.Length * inner;
                for (int r = 0; r < rows.Length; r++)
                {
                    for (long i = 0; i < inner; i++)
                    {
                        Buffer.BlockCopy(source, checked((int)((sourceBase + i * outer + rows[r]) * elementBytes)),
                            result, checked((int)((resultBase + r * inner + i) * elementBytes)), elementBytes);
                    }
                }
            }

            return result;
        });
    }

    /// <summary>The columns <paramref name="columns"/> of a per-expert bias [E, N] as an F32 [E, columns] weight.</summary>
    public static WeightInfo GatherColumns(WeightInfo w, string name, int[] columns)
    {
        long experts = w.Shape[0].Fixed;
        long width = w.Shape[1].Fixed;
        return WeightInfo.Derived(name, DataType.F32, new Shape(experts, columns.Length), () =>
        {
            var values = HfTransforms.DecodeF32(w);
            var result = new float[experts * columns.Length];
            for (long e = 0; e < experts; e++)
            {
                for (int c = 0; c < columns.Length; c++)
                {
                    result[e * columns.Length + c] = values[e * width + columns[c]];
                }
            }

            return HfTransforms.EncodeF32(result);
        });
    }

    /// <summary>
    /// The MXFP4 [E, rows, in] weight of gpt-oss <c>*_blocks</c> [E, R, in/32, 16] (element 2i in the low nibble of byte i, 2i+1 in the
    /// high one) and <c>*_scales</c> [E, R, in/32] (E8M0), keeping the rows in <paramref name="rowOrder"/>. Each 32-value group
    /// becomes one MXFP4 block: the scale byte, then 16 bytes with value j in the low nibble and value j + 16 in the high one.
    /// </summary>
    public static WeightInfo Mxfp4(WeightInfo blocks, WeightInfo scales, string name, int[]? rowOrder = null)
    {
        long experts = blocks.Shape[0].Fixed;
        long sourceRows = blocks.Shape[1].Fixed;
        long groups = blocks.Shape[2].Fixed;
        if (blocks.Shape[3].Fixed != 16 || scales.Shape[2].Fixed != groups)
        {
            throw new InvalidDataException($"{name}: MXFP4 blocks {blocks.Shape} do not match scales {scales.Shape}.");
        }

        int[] rows = rowOrder ?? [.. Enumerable.Range(0, checked((int)sourceRows))];
        return WeightInfo.Derived(name, DataType.MXFP4, new Shape(experts, rows.Length, groups * 32), () =>
        {
            var packed = blocks.ReadAll();
            var scale = scales.ReadAll();
            var result = new byte[experts * rows.Length * groups * 17];
            Span<byte> values = stackalloc byte[32];
            long target = 0;
            for (long e = 0; e < experts; e++)
            {
                for (int r = 0; r < rows.Length; r++)
                {
                    long sourceRow = e * sourceRows + rows[r];
                    for (long g = 0; g < groups; g++)
                    {
                        long group = sourceRow * groups + g;
                        for (int i = 0; i < 16; i++)
                        {
                            byte b = packed[group * 16 + i];
                            values[2 * i] = (byte)(b & 0x0F);
                            values[2 * i + 1] = (byte)(b >> 4);
                        }

                        result[target++] = scale[group];
                        for (int j = 0; j < 16; j++)
                        {
                            result[target++] = (byte)(values[j] | (values[j + 16] << 4));
                        }
                    }
                }
            }

            return result;
        });
    }
}
