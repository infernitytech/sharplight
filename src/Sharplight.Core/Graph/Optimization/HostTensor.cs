using Sharplight.Tensors;

namespace Sharplight.Graph.Optimization;

/// <summary>A small dense row-major tensor in host memory, as evaluated by <see cref="ConstantEvaluator"/>.</summary>
internal sealed class HostTensor(DataType type, long[] dims, byte[] data)
{
    public DataType Type { get; } = type;
    public long[] Dims { get; } = dims;
    public byte[] Data { get; } = data;
    public long Elements => Dims.Aggregate(1L, (a, b) => a * b);

    /// <summary>Wraps a constant value; <see langword="null"/> when its shape is not fully static or its bytes do not match it.</summary>
    public static HostTensor? Of(Value v)
    {
        if (v.Kind != ValueKind.Constant || v.ConstantData is not { } data || !GraphFacts.IsFullyStatic(v.Shape) || v.Type.IsQuantized)
        {
            return null;
        }

        var dims = v.Shape.Dims.Select(d => d.Fixed).ToArray();
        var t = new HostTensor(v.Type, dims, data);
        return v.Type.ByteSize(t.Elements) == data.Length ? t : null;
    }

    /// <summary>Element <paramref name="i"/> read as f32 (types the reference backend reads as f32: f32, f16, bf16).</summary>
    public float ReadFloat(long i)
    {
        if (Type == DataType.F32)
        {
            return BitConverter.ToSingle(Data, (int)(i * 4));
        }

        ushort bits = BitConverter.ToUInt16(Data, (int)(i * 2));
        return Type == DataType.F16 ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.Int32BitsToSingle(bits << 16);
    }

    /// <summary>Whether <see cref="ReadFloat"/> supports the type.</summary>
    public static bool IsFloatLike(DataType t) => t == DataType.F32 || t == DataType.F16 || t == DataType.BF16;
}
