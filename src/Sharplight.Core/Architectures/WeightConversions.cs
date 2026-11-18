using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>Reads small f32, f16 or bf16 weights as floats, for adapters that fold or widen them while building graphs.</summary>
internal static class WeightConversions
{
    /// <summary>True for the float types <see cref="ReadFloats"/> reads.</summary>
    /// <param name="type">A storage type.</param>
    public static bool IsFloat(DataType type) => type == DataType.F32 || type == DataType.F16 || type == DataType.BF16;

    /// <summary>The values of an f32, f16 or bf16 weight.</summary>
    /// <param name="weight">The weight.</param>
    /// <exception cref="NotSupportedException">The weight has another type.</exception>
    public static float[] ReadFloats(WeightInfo weight)
    {
        byte[] bytes = weight.ReadAll();
        if (weight.Type == DataType.F32)
        {
            return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        }

        if (!IsFloat(weight.Type))
        {
            throw new NotSupportedException($"{weight.Name}: {weight.Type} cannot be read as floats.");
        }

        var halves = MemoryMarshal.Cast<byte, ushort>(bytes);
        var values = new float[halves.Length];
        bool f16 = weight.Type == DataType.F16;
        for (int i = 0; i < halves.Length; i++)
        {
            values[i] = f16 ? (float)BitConverter.UInt16BitsToHalf(halves[i]) : BitConverter.Int32BitsToSingle(halves[i] << 16);
        }

        return values;
    }

    /// <summary><paramref name="count"/> values of an f32, f16 or bf16 weight starting at element <paramref name="first"/> (a row of a table), read without loading the rest.</summary>
    /// <param name="weight">The weight.</param>
    /// <param name="first">The first element.</param>
    /// <param name="count">The number of elements.</param>
    /// <exception cref="NotSupportedException">The weight has another type.</exception>
    public static float[] ReadFloats(WeightInfo weight, long first, long count)
    {
        var values = new float[count];
        if (weight.Type == DataType.F32)
        {
            weight.Source.Read(first * sizeof(float), MemoryMarshal.AsBytes(values.AsSpan()));
            return values;
        }

        if (!IsFloat(weight.Type))
        {
            throw new NotSupportedException($"{weight.Name}: {weight.Type} cannot be read as floats.");
        }

        var raw = new ushort[count];
        weight.Source.Read(first * sizeof(ushort), MemoryMarshal.AsBytes(raw.AsSpan()));
        bool f16 = weight.Type == DataType.F16;
        for (int i = 0; i < raw.Length; i++)
        {
            values[i] = f16 ? (float)BitConverter.UInt16BitsToHalf(raw[i]) : BitConverter.Int32BitsToSingle(raw[i] << 16);
        }

        return values;
    }

    /// <summary>An f16 or bf16 weight widened to f32 on first read, under a new name and shape (same element count).</summary>
    /// <param name="weight">The weight.</param>
    /// <param name="name">The new name.</param>
    /// <param name="shape">The new shape.</param>
    public static WeightInfo ToF32(WeightInfo weight, string name, Shape shape)
        => WeightInfo.Derived(name, DataType.F32, shape, () => MemoryMarshal.AsBytes(ReadFloats(weight).AsSpan()).ToArray());
}
