using System.Buffers.Binary;
using Sharplight.Tensors;
using Sharplight.Tensors.Grids;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The lookup grids of the IQ weight formats packed into device constants. Kernels read them through a pointer that the lowering prepends to
/// the arguments of every launch that touches such a weight (a table in kernel-local memory is initialised per thread and indexed divergently,
/// which makes the decoders several times slower than a table in cached global memory).
/// </summary>
/// <remarks>
/// Layout in 32-bit words, mirrored by <c>quant_iq.slang</c>: the grid (a 64-bit entry is two words, low half first; IQ1 entries are one word),
/// then for the formats that use 7-bit sign indices the 128 sign bytes (<see cref="IqSigns"/>) as 32 words.
/// </remarks>
internal static class GridTables
{
    /// <summary>The key of the grid buffer a format reads, or null when its decoder needs no grid.</summary>
    public static string? KeyOf(DataType type)
    {
        if (type == DataType.IQ2_XXS)
        {
            return "iq2_xxs";
        }

        if (type == DataType.IQ2_XS)
        {
            return "iq2_xs";
        }

        if (type == DataType.IQ2_S)
        {
            return "iq2_s";
        }

        if (type == DataType.IQ3_XXS)
        {
            return "iq3_xxs";
        }

        if (type == DataType.IQ3_S)
        {
            return "iq3_s";
        }

        if (type == DataType.IQ1_S || type == DataType.IQ1_M)
        {
            return "iq1";
        }

        return null;
    }

    /// <summary>Builds the constant buffer content for a key returned by <see cref="KeyOf"/>.</summary>
    public static byte[] Build(string key) => key switch
    {
        "iq2_xxs" => Pack(WideEntries(Iq2XxsGrid.Values), withSigns: true),
        "iq2_xs" => Pack(WideEntries(Iq2XsGrid.Values), withSigns: true),
        "iq2_s" => Pack(WideEntries(Iq2SGrid.Values), withSigns: false),
        "iq3_xxs" => Pack([.. Iq3XxsGrid.Values], withSigns: true),
        "iq3_s" => Pack([.. Iq3SGrid.Values], withSigns: false),
        "iq1" => Pack(CompactIq1Entries(), withSigns: false),
        _ => throw new ArgumentOutOfRangeException(nameof(key), key, "unknown grid"),
    };

    private static uint[] WideEntries(ReadOnlySpan<ulong> grid)
    {
        var words = new uint[grid.Length * 2];
        for (int i = 0; i < grid.Length; i++)
        {
            words[2 * i] = (uint)grid[i];
            words[2 * i + 1] = (uint)(grid[i] >> 32);
        }

        return words;
    }

    /// <summary>IQ1 entries as ggml's GPU table stores them: value + 1 per nibble, values 0..3 in the low nibbles of the four bytes, values 4..7 in the high nibbles.</summary>
    private static uint[] CompactIq1Entries()
    {
        var grid = Iq1SGrid.Values;
        var words = new uint[grid.Length];
        for (int i = 0; i < grid.Length; i++)
        {
            uint word = 0;
            for (int j = 0; j < 4; j++)
            {
                uint low = (uint)((sbyte)(grid[i] >> (8 * j)) + 1);
                uint high = (uint)((sbyte)(grid[i] >> (8 * (j + 4))) + 1);
                word |= low << (8 * j);
                word |= high << (8 * j + 4);
            }

            words[i] = word;
        }

        return words;
    }

    private static byte[] Pack(uint[] grid, bool withSigns)
    {
        var signs = IqSigns.Values;
        var bytes = new byte[(grid.Length + (withSigns ? signs.Length / 4 : 0)) * 4];
        for (int i = 0; i < grid.Length; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4), grid[i]);
        }

        if (withSigns)
        {
            signs.CopyTo(bytes.AsSpan(grid.Length * 4));
        }

        return bytes;
    }
}
