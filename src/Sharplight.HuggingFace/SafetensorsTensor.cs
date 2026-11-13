using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>One tensor entry of a safetensors header.</summary>
internal sealed class SafetensorsTensor
{
    /// <summary>The tensor name as stored in the checkpoint (Hugging Face naming).</summary>
    public required string Name { get; init; }

    /// <summary>The safetensors dtype string, e.g. "BF16".</summary>
    public required string DType { get; init; }

    /// <summary>Sharplight type, or null if the dtype has no Sharplight equivalent (reading it throws).</summary>
    public DataType? Type { get; init; }

    /// <summary>Shape, outermost dimension first (same convention as <see cref="Tensors.Shape"/>).</summary>
    public required long[] Shape { get; init; }

    /// <summary>Offset relative to the start of the data section.</summary>
    public required long Offset { get; init; }

    /// <summary>Size of the tensor data in bytes.</summary>
    public required long ByteLength { get; init; }

    /// <summary>The shard this tensor lives in.</summary>
    public required SafetensorsFile File { get; init; }

    /// <summary>A zero-copy weight view over the memory map.</summary>
    /// <exception cref="NotSupportedException">The dtype has no Sharplight equivalent.</exception>
    public WeightInfo ToWeightInfo()
    {
        var type = Type ?? throw new NotSupportedException(
            $"{File.Path}: tensor '{Name}' has unsupported safetensors dtype '{DType}'.");
        return new WeightInfo
        {
            Name = Name,
            Type = type,
            Shape = new Shape(Shape.Select(d => (Dim)d)),
            Source = new SafetensorsWeightSource(File, File.DataOffset + Offset, ByteLength),
        };
    }

    /// <summary>Maps a safetensors dtype to a Sharplight type, or null when it is parsed but not readable.</summary>
    public static DataType? MapDType(string dtype) => dtype switch
    {
        "F32" => DataType.F32,
        "F16" => DataType.F16,
        "BF16" => DataType.BF16,
        "F64" => DataType.F64,
        "I8" => DataType.I8,
        "U8" => DataType.U8,
        "I16" => DataType.I16,
        "I32" => DataType.I32,
        "U32" => DataType.U32,
        "I64" => DataType.I64,
        "BOOL" => DataType.Bool,
        _ => null, // F8_*, U16, U64, F4, ... : parsed but not readable
    };

    /// <summary>Bytes per element of a safetensors dtype, or 0 for unknown / sub-byte types (their size is not validated).</summary>
    public static int ElementBytes(string dtype) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        "I8" or "U8" or "BOOL" or "F8_E4M3" or "F8_E5M2" or "F8_E8M0" => 1,
        _ => 0,
    };
}
