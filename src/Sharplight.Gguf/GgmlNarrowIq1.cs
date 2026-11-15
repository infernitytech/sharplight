using System.Buffers.Binary;

namespace Sharplight.Gguf;

/// <summary>
/// The narrow IQ1 types of unsloth's llama.cpp fork (ggml ids 64-66), which keep IQ1_S's 8-value ternary codebook, f16 super-block
/// scale, 3-bit sub-block scales and delta signs but index a subset of the codebook (<see cref="GgmlNarrowIq1Grids"/>) with fewer bits.
/// They are read as IQ1_S: every 256-value block widens to the IQ1_S block with the same values, so the IQ1_S kernels run them.
/// </summary>
/// <remarks>
/// Block layouts (little endian), per 256 values; sub-block ib (32 values) has four groups of 8, group l of sub-block ib indexes
/// <c>qs[4·ib + l]</c> plus high bits:
/// <list type="bullet">
/// <item>IQ1_XS (64, 46 bytes): f16 d, u8 qs[32], u8 qh[8] (high bits of group l: <c>(qh[ib] &gt;&gt; 2l) &amp; 3</c>), u8 sc[4] (nibble per
/// sub-block, low nibble first: bits 0-2 scale, bit 3 delta sign).</item>
/// <item>IQ1_XXS (65, 42 bytes): f16 d, u8 qs[32], u8 qh[8] (bit l: high bit of group l, bits 4-6 scale, bit 7 delta sign).</item>
/// <item>IQ1_XXXS (66, 38 bytes): f16 d, u8 qs[32] (the whole index), u8 sc[4] as IQ1_XS.</item>
/// </list>
/// IQ1_S (50 bytes): f16 d, u8 qs[32], u16 qh[8] (bits 3l..3l+2: high bits of group l, bits 12-14 scale, bit 15 delta sign).
/// </remarks>
internal static class GgmlNarrowIq1
{
    /// <summary>ggml id of IQ1_XS (1.4375 bpw, 1024-point codebook).</summary>
    public const int Iq1Xs = 64;

    /// <summary>ggml id of IQ1_XXS (1.3125 bpw, 512-point codebook).</summary>
    public const int Iq1Xxs = 65;

    /// <summary>ggml id of IQ1_XXXS (1.1875 bpw, 256-point codebook).</summary>
    public const int Iq1Xxxs = 66;

    /// <summary>Bytes of a widened (IQ1_S) block.</summary>
    public const int WideBlockBytes = 50;

    private const int Groups = 32;
    private const int SubBlocks = 8;
    private const int ScaleBytes = 2;

    /// <summary>Whether <paramref name="ggmlType"/> is one of the narrow IQ1 ids.</summary>
    public static bool Is(int ggmlType) => ggmlType is Iq1Xs or Iq1Xxs or Iq1Xxxs;

    /// <summary>Stored bytes of one 256-value block of a narrow IQ1 type.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The id is not a narrow IQ1 type.</exception>
    public static int StoredBlockBytes(int ggmlType) => ggmlType switch
    {
        Iq1Xs => 46,
        Iq1Xxs => 42,
        Iq1Xxxs => 38,
        _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
    };

    /// <summary>Writes the IQ1_S block (<see cref="WideBlockBytes"/>) holding the values of one narrow block.</summary>
    /// <param name="ggmlType">The narrow type id.</param>
    /// <param name="narrow">The stored block (<see cref="StoredBlockBytes"/> bytes).</param>
    /// <param name="wide">The IQ1_S block to write.</param>
    public static void Widen(int ggmlType, ReadOnlySpan<byte> narrow, Span<byte> wide)
    {
        var grid = ggmlType switch
        {
            Iq1Xs => GgmlNarrowIq1Grids.Xs,
            Iq1Xxs => GgmlNarrowIq1Grids.Xxs,
            _ => GgmlNarrowIq1Grids.Xxxs,
        };
        narrow[..ScaleBytes].CopyTo(wide);
        var qs = narrow.Slice(ScaleBytes, Groups);
        var tail = narrow[(ScaleBytes + Groups)..];
        for (int ib = 0; ib < SubBlocks; ib++)
        {
            int scaleAndSign = ggmlType == Iq1Xxs ? tail[ib] >> 4 : (tail[ggmlType == Iq1Xs ? SubBlocks + ib / 2 : ib / 2] >> (4 * (ib & 1))) & 0xf;
            int qh = (scaleAndSign & 7) << 12 | (scaleAndSign >> 3) << 15;
            for (int l = 0; l < 4; l++)
            {
                int high = ggmlType switch
                {
                    Iq1Xs => (tail[ib] >> (2 * l)) & 3,
                    Iq1Xxs => (tail[ib] >> l) & 1,
                    _ => 0,
                };
                int index = grid[qs[4 * ib + l] | high << 8];
                wide[ScaleBytes + 4 * ib + l] = (byte)index;
                qh |= (index >> 8) << (3 * l);
            }

            BinaryPrimitives.WriteUInt16LittleEndian(wide[(ScaleBytes + Groups + 2 * ib)..], (ushort)qh);
        }
    }
}
