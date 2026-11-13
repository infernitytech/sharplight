using Sharplight.Tensors;

namespace Sharplight.Graph;

// Pooling variants beyond Pool: max-pool argmax indices and region-of-interest max pooling.
public static partial class StandardOps
{
    /// <summary>
    /// Positions of the maxima of a 2-D max pooling over x [N, C, H, W] (the attributes of <see cref="Pool"/> with mode max) → i32
    /// [N, C, HO, WO] flat indices into x: <c>((n·C + c)·H + h)·W + w</c>, or <c>((n·C + c)·W + w)·H + h</c> with <c>storage_order</c> 1.
    /// Ties resolve to the first maximum in row-major window order.
    /// </summary>
    public static readonly OpSchema PoolArgMax = new("PoolArgMax", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.I32, PooledShape(i[0].Shape, a))]);

    /// <summary>
    /// Region-of-interest max pooling (Fast R-CNN): x [N, C, H, W], rois [R, 5] rows (batch index, x1, y1, x2, y2) in input coordinates
    /// scaled by <c>spatial_scale</c> and rounded → [R, C, PH, PW] with <c>pooled_shape</c> long[2] = (PH, PW). Empty bins are 0.
    /// </summary>
    public static readonly OpSchema MaxRoiPool = new("MaxRoiPool", OpCategory.Data, 2, 2, (i, a) =>
    {
        var pooled = a.Get<long[]>("pooled_shape");
        return [new(DataType.F32, new Shape(i[1].Shape[0], i[0].Shape[1], pooled[0], pooled[1]))];
    });
}
