namespace Sharplight.HuggingFace;

/// <summary>
/// How the scales of an FP8 weight [batch..., N, K] cover its values: the scale of value (b, n, k) is
/// <c>scales[b * BatchStride + (n / RowBlock) * Columns + k / ColumnBlock]</c>. Per-tensor, per-matrix, per-channel, group and
/// block (DeepSeek 128 x 128) scales are all instances.
/// </summary>
/// <param name="BatchStride">Scales per leading index (0: shared by all matrices of a stacked tensor).</param>
/// <param name="RowBlock">Rows sharing a scale.</param>
/// <param name="ColumnBlock">Columns (K values) sharing a scale; K or more for one scale per row.</param>
/// <param name="Columns">Scales per scale row.</param>
internal readonly record struct HfFp8ScaleLayout(long BatchStride, long RowBlock, long ColumnBlock, long Columns)
{
    /// <summary>The scale of value (<paramref name="batch"/>, <paramref name="row"/>, <paramref name="column"/>).</summary>
    public float At(float[] scales, long batch, long row, long column)
        => scales[batch * BatchStride + row / RowBlock * Columns + column / ColumnBlock];

    /// <summary>
    /// The layout of a scale tensor of shape <paramref name="scaleShape"/> for <paramref name="batch"/> matrices of
    /// <paramref name="rows"/> x <paramref name="columns"/>; null when the shapes do not fit any known layout.
    /// </summary>
    /// <param name="batch">Product of the leading weight dimensions (1 for a matrix).</param>
    /// <param name="rows">Weight rows N.</param>
    /// <param name="columns">Weight columns K.</param>
    /// <param name="scaleShape">Shape of the scale tensor (empty for a scalar).</param>
    /// <param name="configBlock">The checkpoint's declared block size [rows, columns] (<c>weight_block_size</c>), if any.</param>
    public static HfFp8ScaleLayout? Infer(long batch, long rows, long columns, long[] scaleShape, (long Rows, long Columns)? configBlock)
    {
        long elements = scaleShape.Aggregate(1L, (a, d) => a * d);
        if (elements == 1)
        {
            return new HfFp8ScaleLayout(0, rows, columns, 1);
        }

        if (batch > 1 && elements == batch)
        {
            return new HfFp8ScaleLayout(1, rows, columns, 1);
        }

        bool perRow = scaleShape.Length == 1 || scaleShape[^1] == 1;
        if (elements == batch * rows && perRow)
        {
            return new HfFp8ScaleLayout(rows, 1, columns, 1);
        }

        if (scaleShape.Length < 2)
        {
            return null;
        }

        long scaleRows = scaleShape[^2];
        long scaleColumns = scaleShape[^1];
        if (batch * scaleRows * scaleColumns != elements)
        {
            return null;
        }

        long? rowBlock = BlockSize(rows, scaleRows, configBlock?.Rows);
        long? columnBlock = BlockSize(columns, scaleColumns, configBlock?.Columns);
        if (rowBlock is null || columnBlock is null)
        {
            return null;
        }

        return new HfFp8ScaleLayout(scaleRows * scaleColumns, rowBlock.Value, columnBlock.Value, scaleColumns);
    }

    /// <summary>
    /// The extent of one scale along a dimension of <paramref name="length"/> with <paramref name="scales"/> scales: the declared
    /// block size when it fits, else the exact quotient, else the smallest power of two that gives that many blocks.
    /// </summary>
    private static long? BlockSize(long length, long scales, long? declared)
    {
        static long Blocks(long length, long block) => (length + block - 1) / block;
        if (declared is { } d && d > 0 && Blocks(length, d) == scales)
        {
            return d;
        }

        if (length % scales == 0)
        {
            return length / scales;
        }

        for (long block = 1; block <= length; block *= 2)
        {
            if (Blocks(length, block) == scales)
            {
                return block;
            }
        }

        return null;
    }
}
