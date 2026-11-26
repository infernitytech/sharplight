namespace Sharplight.ImageGeneration;

/// <summary>One tile along one axis of a <see cref="SpatialTiling"/> layout: where it starts and how long it is, in the input's pixels.</summary>
/// <param name="Start">The first pixel the tile covers.</param>
/// <param name="Length">The pixels it covers.</param>
public readonly record struct TileSpan(int Start, int Length);
