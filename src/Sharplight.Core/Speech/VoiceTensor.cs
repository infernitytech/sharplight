using System.Runtime.InteropServices;

namespace Sharplight.Speech;

/// <summary>One named array of a <see cref="VoiceProfile"/>: raw little-endian values with their element type and shape.</summary>
/// <param name="DataType">The element type: <c>f32</c> or <c>i32</c>.</param>
/// <param name="Shape">The dimensions (a single dimension for a flat array).</param>
/// <param name="Data">The values as raw bytes.</param>
public sealed record VoiceTensor(string DataType, int[] Shape, byte[] Data)
{
    /// <summary>The type name of 32-bit floats.</summary>
    public const string Float32 = "f32";

    /// <summary>The type name of 32-bit integers.</summary>
    public const string Int32 = "i32";

    /// <summary>A tensor of floats.</summary>
    /// <param name="values">The values.</param>
    /// <param name="shape">The dimensions, or <see langword="null"/> for a flat array.</param>
    public static VoiceTensor From(float[] values, int[]? shape = null)
        => new(Float32, shape ?? [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    /// <summary>A tensor of integers.</summary>
    /// <param name="values">The values.</param>
    /// <param name="shape">The dimensions, or <see langword="null"/> for a flat array.</param>
    public static VoiceTensor From(int[] values, int[]? shape = null)
        => new(Int32, shape ?? [values.Length], MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    /// <summary>The values as floats.</summary>
    /// <exception cref="InvalidOperationException">The tensor holds another type.</exception>
    public float[] ToFloats()
        => DataType == Float32 ? MemoryMarshal.Cast<byte, float>(Data).ToArray() : throw new InvalidOperationException($"The tensor holds {DataType}, not {Float32}.");

    /// <summary>The values as integers.</summary>
    /// <exception cref="InvalidOperationException">The tensor holds another type.</exception>
    public int[] ToInts()
        => DataType == Int32 ? MemoryMarshal.Cast<byte, int>(Data).ToArray() : throw new InvalidOperationException($"The tensor holds {DataType}, not {Int32}.");
}
