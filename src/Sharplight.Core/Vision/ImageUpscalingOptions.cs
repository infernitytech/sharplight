namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IImageUpscaler"/>.</summary>
public sealed record ImageUpscalingOptions
{
    /// <summary>
    /// The side in input pixels of the tiles a large image is cut into (the memory of the network grows with the tile), or 0 to run the whole image at once.
    /// Images that fit one tile are run whole.
    /// </summary>
    public int TileSize { get; init; } = 192;

    /// <summary>
    /// The context in input pixels each tile shares with its neighbours: the tiles overlap by twice this, and the overlap is blended with linear weights, so the
    /// seams are invisible. More padding is closer to running the whole image at once.
    /// </summary>
    public int TilePadding { get; init; } = 16;
}
