using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>A full weight difference (LyCORIS' <c>diff</c>), added as stored.</summary>
internal sealed class FullDelta(WeightInfo difference) : LoraDelta
{
    /// <inheritdoc/>
    public override long Rows => difference.Shape[0].Fixed;

    /// <inheritdoc/>
    public override long Columns => Inner(difference);

    /// <inheritdoc/>
    public override long? Rank => null;

    /// <inheritdoc/>
    public override string Kind => "full";

    /// <inheritdoc/>
    public override float[] Compute() => DiffusionLoraPatch.Decode(difference);
}
