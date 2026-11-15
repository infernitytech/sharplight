using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// FP-Quant checkpoints (IST-DASLab, <c>quant_method fp_quant</c>, MXFP4 or NVFP4 weights with a Hadamard rotation per group):
/// <c>X.qweight</c> [N, K/2] U8 (two E2M1 codes per byte, the low nibble first), <c>X.scales</c> [N, K/32] E8M0 (MXFP4) or
/// [N, K/16] E4M3 (NVFP4), <c>X.weight_global_scale</c> and the Hadamard matrices become a dense BF16 <c>X.weight</c>.
/// </summary>
/// <remarks>
/// The stored weights live in the rotated domain: the library rotates each activation group by <c>forward_hadamard_matrix</c> H
/// (<c>x_rot[g, j] = sum_i x[g, i] H[i, j]</c>), quantizes it too, and multiplies by <c>1 / (weight_global_scale * act_global_scale)</c>.
/// This importer folds the rotation into the weight, <c>W[o, g, i] = sum_j H[i, j] * dq[o, g, j]</c> with
/// <c>dq = e2m1 * scale / weight_global_scale</c>, so activations are neither rotated nor quantized (more accurate than the
/// library's FP4 activations, and no per-layer rotation op is needed). This is a dequantize-at-import fallback: the BF16 result is
/// not stored as FP4 anymore. Rows are decoded on demand from the memory-mapped tensors.
/// </remarks>
internal static class HfFpQuantWeights
{
    private static readonly float[] s_e2m1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    /// <summary>The checkpoint tensors with FP-Quant modules replaced by dense weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">A module has an unreadable layout.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var modules = tensors.Keys.Where(name => name.EndsWith(".qweight", StringComparison.Ordinal))
            .Select(name => name[..^".qweight".Length])
            .Where(module => tensors.ContainsKey(module + ".scales") && tensors.ContainsKey(module + ".forward_hadamard_matrix")
                && tensors.ContainsKey(module + ".weight_global_scale"))
            .ToList();
        if (modules.Count == 0)
        {
            return tensors;
        }

        bool nvfp4 = config.TryGetQuantization("quantization_config", out var settings)
            && settings.TryGetProperty("forward_dtype", out var dtype) && dtype.GetString() == "nvfp4";
        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (string module in modules)
        {
            var codes = tensors[module + ".qweight"];
            var scales = tensors[module + ".scales"];
            var matrix = tensors[module + ".forward_hadamard_matrix"];
            var globalScale = tensors[module + ".weight_global_scale"];
            foreach (string suffix in (string[])[".qweight", ".scales", ".forward_hadamard_matrix", ".backward_hadamard_matrix", ".weight_global_scale", ".act_global_scale", ".dqweight"])
            {
                result.Remove(module + suffix);
            }

            result[module + ".weight"] = Convert(module + ".weight", codes, scales, matrix, globalScale, nvfp4);
        }

        return result;
    }

    private static SafetensorsTensor Convert(string name, SafetensorsTensor codes, SafetensorsTensor scales, SafetensorsTensor matrix, SafetensorsTensor globalScale, bool nvfp4)
    {
        int block = nvfp4 ? 16 : 32;
        long rows = codes.Shape[0];
        long columns = codes.Shape[1] * 2;
        if (codes.Shape.Length != 2 || codes.DType != "U8" || scales.DType != "U8" || scales.Shape.Length != 2 || scales.Shape[0] != rows
            || scales.Shape[1] * block != columns || matrix.Shape.Length != 2 || matrix.Shape[0] != matrix.Shape[1]
            || matrix.DType is not ("BF16" or "F16" or "F32") || columns % matrix.Shape[0] != 0 || matrix.Shape[0] % block != 0)
        {
            throw new NotSupportedException($"{name}: FP-Quant tensors of shapes qweight [{string.Join(", ", codes.Shape)}], scales [{string.Join(", ", scales.Shape)}], rotation [{string.Join(", ", matrix.Shape)}] are not a known layout.");
        }

        var rotation = HfTransforms.DecodeF32(matrix.ToWeightInfo());
        float inverseGlobal = 1f / globalScale.Scalar();
        var shape = new Shape([(Dim)rows, (Dim)columns]);
        var decoder = new Decoder(codes, scales, rotation, (int)matrix.Shape[0], block, nvfp4, inverseGlobal, columns);
        var converted = new WeightInfo
        {
            Name = name,
            Type = DataType.BF16,
            Shape = shape,
            Source = new HfRowConvertedSource(rows, columns * 2, decoder.DecodeRow),
        };
        return new SafetensorsTensor
        {
            Name = name,
            DType = "BF16",
            Type = DataType.BF16,
            Shape = [rows, columns],
            Offset = codes.Offset,
            ByteLength = converted.ByteSize,
            File = codes.File,
            Converted = converted,
        };
    }

    private sealed unsafe class Decoder(SafetensorsTensor codes, SafetensorsTensor scales, float[] rotation, int group, int block, bool nvfp4, float inverseGlobal, long columns)
    {
        public void DecodeRow(long row, Span<byte> destination)
        {
            Span<float> dq = stackalloc float[group];
            byte* packed = codes.Pointer + row * (columns / 2);
            byte* scaleRow = scales.Pointer + row * (columns / block);
            for (long g = 0; g < columns / group; g++)
            {
                for (int j = 0; j < group; j++)
                {
                    long column = g * group + j;
                    byte pair = packed[column / 2];
                    float value = s_e2m1[(column & 1) == 0 ? pair & 15 : pair >> 4];
                    float scale = nvfp4 ? E4M3(scaleRow[column / block]) : MathF.ScaleB(1f, scaleRow[column / block] - 127);
                    dq[j] = value * scale * inverseGlobal;
                }

                for (int i = 0; i < group; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < group; j++)
                    {
                        sum += rotation[i * group + j] * dq[j];
                    }

                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)((g * group + i) * 2)..], ToBf16(sum));
                }
            }
        }

        private static float E4M3(byte code)
        {
            int exponent = (code >> 3) & 15;
            int mantissa = code & 7;
            if (exponent == 15 && mantissa == 7)
            {
                return float.NaN;
            }

            float magnitude = exponent == 0 ? mantissa / 8f * MathF.ScaleB(1f, -6) : (1f + mantissa / 8f) * MathF.ScaleB(1f, exponent - 7);
            return (code & 0x80) != 0 ? -magnitude : magnitude;
        }

        private static ushort ToBf16(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            if (float.IsNaN(value))
            {
                return 0x7FC0;
            }

            return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
        }
    }
}
