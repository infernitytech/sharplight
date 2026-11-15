using Sharplight.Tensors;

namespace Sharplight.Cpu;

// The trellis-coded formats (DataType.TrellisV4K8_G128 and its siblings), decoded by the shared TrellisBlocks.
internal static unsafe partial class Quant
{
    /// <summary>The trellis types in the order of their <see cref="WeightFormat"/> values (from <see cref="WeightFormat.TrellisV4K8"/>).</summary>
    private static readonly DataType[] s_trellisTypes =
    [
        DataType.TrellisV4K8_G128, DataType.TrellisV4K6_G128, DataType.TrellisV4K7_G128,
        DataType.TrellisV2K5_G128, DataType.TrellisV2K6_G128, DataType.TrellisV2K7_G128,
    ];

    private static DataType TrellisType(WeightFormat f) => s_trellisTypes[f - WeightFormat.TrellisV4K8];

    private static void Trellis(DataType type, byte* p, long nb, float* dst)
    {
        int bytes = type.BlockBytes;
        for (long b = 0; b < nb; b++, p += bytes, dst += TrellisBlocks.BlockSize)
        {
            TrellisBlocks.DecodeBlock(type, new ReadOnlySpan<byte>(p, bytes), new Span<float>(dst, TrellisBlocks.BlockSize));
        }
    }
}
