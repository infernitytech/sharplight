namespace Sharplight.Cpu;

/// <summary>
/// The channels-last (NHWC) layout of rank-4 [N, C, H, W] f32 values: the same dims as NCHW, element strides
/// [H·W·C, 1, W·C, C]. Ops that understand it read and write it directly; every other op sees an ordinary strided view
/// (and materializes it when it needs dense input), so a value in this layout is valid everywhere.
/// </summary>
internal static class ChannelsLast
{
    /// <summary>Channels-last strides of [N, C, H, W] <paramref name="dims"/>.</summary>
    public static long[] Strides(long[] dims) => [dims[1] * dims[2] * dims[3], 1, dims[3] * dims[1], dims[1]];

    /// <summary>True when <paramref name="r"/> is a whole rank-4 f32 tensor in the channels-last layout (with more than one channel).</summary>
    public static bool Is(TRef r) => IsView(r) && PixelStride(r) == r.Dims[1] && RowStride(r) == r.Dims[3] * r.Dims[1]
        && ImageStride(r) == r.Dims[2] * r.Dims[3] * r.Dims[1];

    /// <summary>
    /// True when <paramref name="r"/> is a rank-4 f32 view whose channels are contiguous and whose pixels, rows and images do
    /// not overlap: a channels-last tensor or a slice of one (a channel range, as a Split makes, or a spatial window).
    /// </summary>
    public static bool IsView(TRef r)
    {
        if (r.Rank != 4 || r.Type != Tensors.DataType.F32 || r.Dims[1] == 1 || r.Strides[1] != 1)
        {
            return false;
        }

        long ps = PixelStride(r);
        long rs = RowStride(r);
        return ps >= r.Dims[1] && rs >= r.Dims[3] * ps && ImageStride(r) >= r.Dims[2] * rs;
    }

    /// <summary>Elements from one pixel of a channels-last view to the next along W.</summary>
    public static long PixelStride(TRef r) => r.Dims[3] > 1 ? r.Strides[3] : r.Dims[1];

    /// <summary>Elements from one row of a channels-last view to the next along H.</summary>
    public static long RowStride(TRef r) => r.Dims[2] > 1 ? r.Strides[2] : r.Dims[3] * PixelStride(r);

    /// <summary>Elements from one image of a channels-last view to the next.</summary>
    public static long ImageStride(TRef r) => r.Sym is not null || r.Dims[0] > 1 ? r.Strides[0] : r.Dims[2] * RowStride(r);
}
