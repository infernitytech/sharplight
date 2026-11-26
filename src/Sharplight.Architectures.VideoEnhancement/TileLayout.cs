using Sharplight.ImageGeneration;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>The tiles an autoencoder stage runs on, in pixels of the padded output frame (one tile of the whole frame when untiled).</summary>
/// <param name="Rows">The tiles along the height.</param>
/// <param name="Columns">The tiles along the width.</param>
internal sealed record TileLayout(IReadOnlyList<TileSpan> Rows, IReadOnlyList<TileSpan> Columns)
{
    /// <summary>The pixels of one tile.</summary>
    public long TileArea => (long)Rows[0].Length * Columns[0].Length;

    /// <summary>The whole frame as one tile.</summary>
    /// <param name="height">The padded height.</param>
    /// <param name="width">The padded width.</param>
    public static TileLayout Whole(int height, int width) => new([new TileSpan(0, height)], [new TileSpan(0, width)]);

    /// <summary>The fewest equal tiles of at most <paramref name="maxArea"/> pixels (see <see cref="SpatialTiling.Fit"/>).</summary>
    /// <param name="height">The padded height.</param>
    /// <param name="width">The padded width.</param>
    /// <param name="maxArea">The largest tile area.</param>
    /// <param name="overlap">The least overlap of neighbouring tiles.</param>
    /// <param name="multiple">The alignment of the tiles (the network's size multiple).</param>
    /// <param name="minSide">The shortest tile side.</param>
    public static TileLayout Fit(int height, int width, long maxArea, int overlap, int multiple, int minSide)
    {
        var (rows, columns) = SpatialTiling.Fit(height, width, maxArea, overlap, multiple, minSide);
        return new TileLayout(rows, columns);
    }

    /// <summary>The fewest equal tiles per axis whose sides are at most <paramref name="side"/> pixels.</summary>
    /// <param name="height">The padded height.</param>
    /// <param name="width">The padded width.</param>
    /// <param name="side">The longest tile side.</param>
    /// <param name="overlap">The least overlap of neighbouring tiles.</param>
    /// <param name="multiple">The alignment of the tiles.</param>
    public static TileLayout WithSide(int height, int width, int side, int overlap, int multiple)
        => new(Axis(height, side, overlap, multiple), Axis(width, side, overlap, multiple));

    /// <inheritdoc/>
    public override string ToString() => Rows.Count * Columns.Count == 1
        ? "untiled"
        : $"{Rows.Count} x {Columns.Count} tiles of {Columns[0].Length} x {Rows[0].Length}";

    private static IReadOnlyList<TileSpan> Axis(int size, int side, int overlap, int multiple)
    {
        for (int count = 1; ; count++)
        {
            var spans = SpatialTiling.Even(size, count, overlap, multiple);
            if (spans[0].Length <= side || spans[0].Length <= 2 * overlap)
            {
                return spans;
            }
        }
    }
}
