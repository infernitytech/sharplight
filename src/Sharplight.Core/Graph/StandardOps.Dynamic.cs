using Sharplight.Tensors;

namespace Sharplight.Graph;

// Ops with symbolic or data-dependent output sizes (see docs/dynamic-shapes.md).
//  * Fill has no inputs; its outer extent may be any symbolic expression (Range / ConstantOfShape over a symbolic size).
//  * NonZero, NonMaxSuppression and Unique produce a data-dependent number of rows: the outer output dimension is the dynamic
//    symbol named by attribute "count_symbol" (created with GraphBuilder.DynamicSymbol); the op writes its value into
//    RunParameters.Symbols when it runs.
public static partial class StandardOps
{
    /// <summary>
    /// Generates <c>y[i, ...] = start + i · delta</c> (constant over the inner axes), without inputs: ONNX Range (delta ≠ 0) and
    /// ConstantOfShape (delta 0) over a possibly symbolic size. Attributes: <c>shape</c> (long[]; entry 0 is ignored when
    /// <c>outer_symbol</c> is set), <c>outer_symbol</c> (string, optional: symbol or expression of the outer dimension),
    /// <c>start</c>, <c>delta</c> (float), <c>type</c> (a <see cref="DataType"/> name, default f32).
    /// </summary>
    public static readonly OpSchema Fill = new("Fill", OpCategory.Data, 0, 0, (i, a) =>
    {
        var shape = a.Get<long[]>("shape");
        var dims = shape.Select(d => (Dim)d).ToArray();
        if (a.Get("outer_symbol", "") is { Length: > 0 } outer)
        {
            dims[0] = Dim.Sym(outer);
        }

        return [new(DataType.Get(a.Get("type", "f32")), new Shape(dims))];
    });

    /// <summary>
    /// Coordinates of the non-zero elements of x (any rank ≥ 1) in row-major order → i32 <c>[count, rank]</c>, the transpose of the ONNX
    /// layout so that the data-dependent count is the outer dimension. Attribute <c>count_symbol</c>: the dynamic symbol of the count.
    /// </summary>
    public static readonly OpSchema NonZero = new("NonZero", OpCategory.Data, 1, 1, (i, a) =>
        [new(DataType.I32, new Shape(Dim.Sym(a.Get<string>("count_symbol")), i[0].Rank))]);

    /// <summary>
    /// ONNX NonMaxSuppression: boxes f32 [B, S, 4], scores f32 [B, C, S] → selected i32 <c>[count, 3]</c> rows (batch, class, box), ordered by
    /// batch, class and descending score. Attributes: <c>count_symbol</c>, <c>max_per_class</c> (long), <c>iou_threshold</c> (float),
    /// <c>score_threshold</c> (float, default -inf: no threshold), <c>center_point_box</c> (bool: boxes are [cx, cy, w, h] instead of corners).
    /// </summary>
    public static readonly OpSchema NonMaxSuppression = new("NonMaxSuppression", OpCategory.Data, 2, 2, (i, a) =>
        [new(DataType.I32, new Shape(Dim.Sym(a.Get<string>("count_symbol")), 3))]);

    /// <summary>
    /// ONNX Unique over the flattened input x → values [count] (x's type), first-occurrence indices i32 [count], inverse indices i32 [n]
    /// (n = elements of x) and counts i32 [count]. Attributes: <c>count_symbol</c>, <c>sorted</c> (bool, default true: ascending values;
    /// otherwise first-occurrence order).
    /// </summary>
    public static readonly OpSchema Unique = new("Unique", OpCategory.Data, 1, 1, (i, a) =>
    {
        var count = Dim.Sym(a.Get<string>("count_symbol"));
        var n = i[0].Shape.Rank == 0 ? Dim.Of(1) : i[0].Shape[0] * i[0].Shape.InnerElements;
        return [new(i[0].Type, new Shape(count)), new(DataType.I32, new Shape(count)), new(DataType.I32, new Shape(n)), new(DataType.I32, new Shape(count))];
    });
}
