using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The floating-point modes of MLX quantization (<c>mx.quantize(mode=...)</c>, no biases): <c>mxfp4</c> (E2M1 codes, U8 E8M0 scale per
/// 32 values), <c>nvfp4</c> (E2M1 codes, U8 E4M3 scale per 16 values, no second-level scale) and <c>mxfp8</c> (E4M3 codes, U8 E8M0
/// scale per 32 values). The codes are the bytes of the U32 <c>X.weight</c> in order (value 2i in the low nibble of byte i), which is
/// the layout of the OCP / ModelOpt checkpoints: the FP4 modes are read by <see cref="HfFp4Weights"/> through U8 views
/// (<see cref="DataType.MXFP4"/>, <see cref="DataType.NVFP4"/>; BF16 when K of an NVFP4 weight is not a multiple of 64), MXFP8
/// becomes <see cref="DataType.F8E4M3_B32"/> with the exact power-of-two scale.
/// </summary>
internal static unsafe class HfMlxFloatWeights
{
    /// <summary>Whether <paramref name="mode"/> is one of the FP modes read here.</summary>
    public static bool IsFloatMode(string mode) => mode is "mxfp4" or "nvfp4" or "mxfp8";

    /// <summary>
    /// Adds the weight <paramref name="stem"/>.weight of an FP mode: FP4 modes as a view pair into <paramref name="fp4"/> (converted
    /// by <see cref="HfFp4Weights.Resolve"/> afterwards), MXFP8 as a converted weight into <paramref name="converted"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">The mode is not read.</exception>
    /// <exception cref="InvalidDataException">The scales do not match the codes.</exception>
    public static void Add(string mode, string stem, SafetensorsTensor codes, SafetensorsTensor scales,
        Dictionary<string, SafetensorsTensor> fp4, Dictionary<string, SafetensorsTensor> converted)
    {
        if (!IsFloatMode(mode))
        {
            throw new NotSupportedException($"{stem}: MLX quantization mode '{mode}' is not supported (affine, mxfp4, nvfp4, mxfp8).");
        }

        long bytesPerRow = codes.Shape[^1] * 4;
        long columns = mode == "mxfp8" ? bytesPerRow : bytesPerRow * 2;
        int group = mode == "nvfp4" ? 16 : 32;
        if (scales.DType != "U8" || scales.Shape[^1] * group != columns || !scales.Shape[..^1].SequenceEqual(codes.Shape[..^1]))
        {
            throw new InvalidDataException($"{stem}: MLX {mode} scales [{string.Join(", ", scales.Shape)}] do not match codes [{string.Join(", ", codes.Shape)}].");
        }

        if (mode == "mxfp8")
        {
            converted[stem + ".weight"] = Mxfp8(stem + ".weight", codes, scales, columns);
            return;
        }

        fp4[stem + ".weight"] = View(stem + ".weight", codes, "U8", [.. codes.Shape[..^1], bytesPerRow]);
        fp4[stem + ".weight_scale"] = View(stem + ".weight_scale", scales, mode == "nvfp4" ? "F8_E4M3" : "U8", scales.Shape);
    }

    private static SafetensorsTensor View(string name, SafetensorsTensor source, string dtype, long[] shape) => new()
    {
        Name = name,
        DType = dtype,
        Type = SafetensorsTensor.MapDType(dtype),
        Shape = shape,
        Offset = source.Offset,
        ByteLength = source.ByteLength,
        File = source.File,
    };

    private static SafetensorsTensor Mxfp8(string name, SafetensorsTensor codes, SafetensorsTensor scales, long columns)
    {
        long[] shape = [.. codes.Shape[..^1], columns];
        long rows = shape.Aggregate(1L, (a, d) => a * d) / columns;
        var type = DataType.F8E4M3_B32;
        byte* codeBase = codes.Pointer;
        byte* scaleBase = scales.Pointer;
        var weight = new WeightInfo
        {
            Name = name,
            Type = type,
            Shape = new Shape(shape.Select(d => (Dim)d)),
            Source = new HfRowConvertedSource(rows, type.ByteSize(columns), (row, destination) =>
            {
                byte* packed = codeBase + row * columns;
                byte* scale = scaleBase + row * (columns / 32);
                for (long b = 0; b < columns / 32; b++)
                {
                    var block = destination.Slice((int)(b * 36), 36);
                    BinaryPrimitives.WriteSingleLittleEndian(block, scale[b] == 0xFF ? float.NaN : MathF.ScaleB(1f, scale[b] - 127));
                    new ReadOnlySpan<byte>(packed + b * 32, 32).CopyTo(block[4..]);
                }
            }),
        };
        return new SafetensorsTensor
        {
            Name = name,
            DType = codes.DType,
            Type = type,
            Shape = shape,
            Offset = codes.Offset,
            ByteLength = weight.ByteSize,
            File = codes.File,
            Converted = weight,
        };
    }
}
