using Sharplight.Tensors;

namespace Sharplight.Graph;

// General-purpose tensor ops for broad model coverage (ONNX and other graph formats).
// Conventions:
//  * Boolean tensors are f32 masks (0 = false, anything else = true); comparison/logic ops output 1/0.
//  * Index tensors are i32 (i64 inputs are accepted where noted and converted by providers).
//  * Axes may be negative (counted from the end). The outermost axis may be symbolic; ops that need a
//    fixed extent along an axis reject a symbolic one in shape inference.
public static partial class StandardOps
{
    /// <summary>y = cond ? a : b with broadcasting over all three inputs (cond is a mask).</summary>
    public static readonly OpSchema Where = new("Where", OpCategory.Elementwise, 3, 3, (i, a) =>
        [new(i[1].Type, BroadcastShapes(BroadcastShapes(i[0].Shape, i[1].Shape), i[2].Shape))]);

    /// <summary>Broadcasts x to attribute <c>shape</c> (long[]; ONNX Expand semantics, i.e. numpy broadcast of both).</summary>
    public static readonly OpSchema Expand = new("Expand", OpCategory.Data, 1, 1, (i, a) =>
    {
        var target = a.Get<long[]>("shape");
        return [new(i[0].Type, BroadcastShapes(i[0].Shape, ShapeWithSymbol(target, i[0].Shape)))];
    });

    /// <summary>Repeats x along each axis: attribute <c>repeats</c> long[rank] (outermost must be 1 if symbolic).</summary>
    public static readonly OpSchema Tile = new("Tile", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        var r = a.Get<long[]>("repeats");
        if (r.Length != x.Rank)
        {
            throw new ArgumentException("Tile: repeats must match rank.");
        }

        var dims = new Dim[x.Rank];
        for (int d = 0; d < x.Rank; d++)
        {
            dims[d] = x[d].IsSymbolic ? (r[d] == 1 ? x[d] : throw new ArgumentException("Tile on a symbolic axis.")) : x[d].Fixed * r[d];
        }

        return [new(i[0].Type, new Shape(dims))];
    });

    /// <summary>
    /// numpy-style take along <c>axis</c>: data [..., A, ...], indices (i32/i64, any shape) →
    /// data.shape[:axis] + indices.shape + data.shape[axis+1:]. Negative indices count from the end.
    /// (<see cref="Gather"/> is the specialised row lookup into weight tables.)
    /// </summary>
    public static readonly OpSchema GatherAxis = new("GatherAxis", OpCategory.Data, 2, 2, (i, a) =>
    {
        var x = i[0].Shape;
        var idx = i[1].Shape;
        int axis = Axis(a.Get("axis", 0L), x.Rank);
        var dims = new List<Dim>();
        for (int d = 0; d < axis; d++)
        {
            dims.Add(x[d]);
        }

        dims.AddRange(idx.Dims);
        for (int d = axis + 1; d < x.Rank; d++)
        {
            dims.Add(x[d]);
        }

        return [new(i[0].Type, new Shape(dims))];
    });

    /// <summary>ONNX GatherElements: out[i][j][k] = data[...index(i,j,k) on axis...]; output has the indices' shape.</summary>
    public static readonly OpSchema GatherElements = new("GatherElements", OpCategory.Data, 2, 2, (i, a) => [new(i[0].Type, i[1].Shape)]);

    /// <summary>ONNX ScatterElements (data, indices, updates; attrs axis, reduction "none"|"add"|"mul"|"max"|"min"). Output = data's shape.</summary>
    public static readonly OpSchema ScatterElements = new("ScatterElements", OpCategory.Data, 3, 3, (i, a) => [new(i[0].Type, i[0].Shape)]);

    /// <summary>ONNX ScatterND (data, indices [..., q], updates; attr reduction). Output = data's shape.</summary>
    public static readonly OpSchema ScatterND = new("ScatterND", OpCategory.Data, 3, 3, (i, a) => [new(i[0].Type, i[0].Shape)]);

    /// <summary>ONNX GatherND (data, indices [..., q]; attr batch_dims).</summary>
    public static readonly OpSchema GatherND = new("GatherND", OpCategory.Data, 2, 2, (i, a) =>
    {
        var x = i[0].Shape;
        var idx = i[1].Shape;
        int b = (int)a.Get("batch_dims", 0L);
        long q = idx[-1].Fixed;
        var dims = new List<Dim>(idx.Dims.Take(idx.Rank - 1));
        for (int d = b + (int)q; d < x.Rank; d++)
        {
            dims.Add(x[d]);
        }

        return [new(i[0].Type, new Shape(dims))];
    });

    /// <summary>Cumulative sum along <c>axis</c>; attrs <c>exclusive</c>, <c>reverse</c> (bool).</summary>
    public static readonly OpSchema CumSum = new("CumSum", OpCategory.Reduction, 1, 1, SameF32);

    /// <summary>
    /// Index of the max/min along <c>axis</c> → i32. Attrs: <c>mode</c> ("max"|"min"), <c>axis</c>,
    /// <c>keepdims</c> (default true), <c>select_last_index</c> (bool).
    /// </summary>
    public static readonly OpSchema ArgReduce = new("ArgReduce", OpCategory.Reduction, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        int axis = Axis(a.Get("axis", 0L), x.Rank);
        return [new(DataType.I32, ReduceShape(x, [axis], a.Get("keepdims", true)))];
    });

    /// <summary>
    /// Top-k along <c>axis</c> (default -1). Attrs: <c>k</c> (long), <c>largest</c> (default true), <c>sorted</c> (default true).
    /// Outputs: values (f32) and indices (i32), both with dim k on the axis.
    /// </summary>
    public static readonly OpSchema TopK = new("TopK", OpCategory.Reduction, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        int axis = Axis(a.Get("axis", -1L), x.Rank);
        var s = x.WithDim(axis, a.Get<long>("k"));
        return [new(DataType.F32, s), new(DataType.I32, s)];
    });

    /// <summary>
    /// Group normalization of x [N, C, *] over (C/groups, *) per group, then per-channel scale/bias [C].
    /// Attrs: <c>groups</c>, <c>eps</c>. InstanceNorm = groups C; LayerNorm over (C,*) = groups 1.
    /// </summary>
    public static readonly OpSchema GroupNorm = new("GroupNorm", OpCategory.Reduction, 1, 3, SameF32);

    /// <summary>
    /// 2-D transposed convolution. x [N, C, H, W], weight [C, O/groups, KH, KW], optional bias [O].
    /// Attrs: strides, pads (top, left, bottom, right), dilations, groups, output_padding (long[2]).
    /// HO = (H-1)·s - (top+bottom) + d·(KH-1) + output_padding + 1.
    /// </summary>
    public static readonly OpSchema ConvTranspose = new("ConvTranspose", OpCategory.MatMul, 2, 3, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        long g = a.Get("groups", 1L);
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        var op = a.Get("output_padding", new long[] { 0, 0 });
        long ho = (x[2].Fixed - 1) * st[0] - pd[0] - pd[2] + dl[0] * (w[2].Fixed - 1) + op[0] + 1;
        long wo = (x[3].Fixed - 1) * st[1] - pd[1] - pd[3] + dl[1] * (w[3].Fixed - 1) + op[1] + 1;
        return [new(DataType.F32, new Shape(x[0], w[1].Fixed * g, ho, wo))];
    });

    /// <summary>Keeps the upper (attr <c>upper</c>, default true) or lower triangle of the last two axes; attr <c>k</c> diagonal offset.</summary>
    public static readonly OpSchema Trilu = new("Trilu", OpCategory.Elementwise, 1, 1, Same);

    /// <summary>One-hot: indices (i32/i64) → f32 with a new axis of attr <c>depth</c> at attr <c>axis</c> (default -1); attrs <c>on</c>, <c>off</c>.</summary>
    public static readonly OpSchema OneHot = new("OneHot", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        int axis = (int)a.Get("axis", -1L);
        if (axis < 0)
        {
            axis += x.Rank + 1;
        }

        var dims = x.Dims.ToList();
        dims.Insert(axis, a.Get<long>("depth"));
        return [new(DataType.F32, new Shape(dims))];
    });

    /// <summary>
    /// Bilinear/nearest sampling of x [N, C, H, W] at normalized grid [N, HO, WO, 2] (x, y in [-1, 1]).
    /// Attrs: <c>mode</c> ("linear"|"nearest"|"cubic"), <c>padding</c> ("zeros"|"border"|"reflection"), <c>align_corners</c>.
    /// </summary>
    public static readonly OpSchema GridSample = new("GridSample", OpCategory.Data, 2, 2, (i, a) =>
    {
        var x = i[0].Shape;
        var g = i[1].Shape;
        return [new(DataType.F32, new Shape(x[0], x[1], g[1], g[2]))];
    });

    internal static int Axis(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);

    internal static Shape ReduceShape(Shape x, IEnumerable<int> axes, bool keep)
    {
        var set = axes.ToHashSet();
        var dims = new List<Dim>();
        for (int d = 0; d < x.Rank; d++)
        {
            if (!set.Contains(d))
            {
                dims.Add(x[d]);
            }
            else if (keep)
            {
                dims.Add(1);
            }
        }

        return new Shape(dims);
    }

    /// <summary>Turns a fixed target shape into a Shape, keeping the input's symbolic outermost dim where the target has 1 or the same extent.</summary>
    private static Shape ShapeWithSymbol(long[] target, Shape like)
    {
        var dims = target.Select(t => (Dim)t).ToArray();
        int off = target.Length - like.Rank;
        if (like.Rank > 0 && like[0].IsSymbolic && off >= 0 && target[off] <= 1)
        {
            dims[off] = like[0];
        }

        return new Shape(dims);
    }
}
