using System.Globalization;

namespace Sharplight.Cuda;

/// <summary>
/// A tensor map a kernel takes in its parameters (see <see cref="Shaders.SlangCompiler.CudaTensorMapsHeader"/>): byte elements over
/// the memory of one of its pointer arguments, encoded at launch recording with that pointer's address.
/// </summary>
/// <param name="Pointer">Index of the pointer argument (in the order of the kernel's pointers).</param>
/// <param name="Dims">Dimensions, innermost first (elements = bytes).</param>
/// <param name="Strides">Byte strides of the dimensions after the innermost.</param>
/// <param name="Box">Box copied per tensor copy, innermost first.</param>
internal sealed record CudaTensorMapSpec(int Pointer, ulong[] Dims, ulong[] Strides, uint[] Box)
{
    /// <summary>Parses the descriptions of a tensor-map header line (without its prefix).</summary>
    /// <exception cref="FormatException">A description is malformed.</exception>
    public static CudaTensorMapSpec[] ParseList(string text) =>
        [.. text.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Parse)];

    private static CudaTensorMapSpec Parse(string description)
    {
        var parts = description.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int rank = parts.Length == 5 ? int.Parse(parts[1], CultureInfo.InvariantCulture) : 0;
        var dims = parts.Length == 5 ? parts[2].Split(',').Select(v => ulong.Parse(v, CultureInfo.InvariantCulture)).ToArray() : [];
        var strides = parts.Length == 5 ? parts[3].Split(',').Select(v => ulong.Parse(v, CultureInfo.InvariantCulture)).ToArray() : [];
        var box = parts.Length == 5 ? parts[4].Split(',').Select(v => uint.Parse(v, CultureInfo.InvariantCulture)).ToArray() : [];
        if (rank < 1 || rank > 5 || dims.Length != rank || strides.Length != rank - 1 || box.Length != rank)
        {
            throw new FormatException($"Malformed tensor map description '{description}'.");
        }

        return new CudaTensorMapSpec(int.Parse(parts[0], CultureInfo.InvariantCulture), dims, strides, box);
    }
}
