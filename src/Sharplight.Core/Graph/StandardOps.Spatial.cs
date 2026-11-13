using Sharplight.Tensors;

namespace Sharplight.Graph;

// Ops for general (non-LLM) networks: CNNs, vision transformers, audio encoders.
// Layout convention for spatial ops is channels-first: x [N, C, *spatial] with N the (possibly
// symbolic) outermost batch axis; spatial dims are fixed. Only 2-D spatial ops are defined for now
// (1-D inputs can be expressed as H=1).
public static partial class StandardOps
{
    /// <summary>y = min(max(x, min), max). Attributes: min (float, default -inf), max (float, default +inf).</summary>
    public static readonly OpSchema Clip = new("Clip", OpCategory.Elementwise, 1, 1, SameF32);

    /// <summary>
    /// 2-D convolution (cross-correlation, as in ONNX/PyTorch).
    /// x [N, C, H, W]; weight [O, C/groups, KH, KW] (any float or quantized weight type); optional bias [O].
    /// Attributes: <c>strides</c> long[2] (default 1,1), <c>pads</c> long[4] = (top, left, bottom, right)
    /// (default 0), <c>dilations</c> long[2] (default 1,1), <c>groups</c> (default 1).
    /// Output [N, O, HO, WO], HO = (H + top + bottom - dil·(KH-1) - 1) / stride + 1.
    /// </summary>
    public static readonly OpSchema Conv = new("Conv", OpCategory.MatMul, 2, 3, (i, a) =>
    {
        var x = i[0].Shape; var w = i[1].Shape;
        if (x.Rank != 4 || w.Rank != 4) throw new ArgumentException($"Conv expects x [N,C,H,W] and w [O,C/g,KH,KW], got x{x} w{w}.");
        long groups = a.Get("groups", 1L);
        if (x[1].Fixed != w[1].Fixed * groups) throw new ArgumentException($"Conv channel mismatch: x{x} w{w} groups={groups}.");
        var (ho, wo) = SpatialOut(x[2].Fixed, x[3].Fixed, w[2].Fixed, w[3].Fixed, a, ceil: false);
        return [new(DataType.F32, new Shape(x[0], w[0], ho, wo))];
    });

    /// <summary>
    /// 2-D pooling over x [N, C, H, W]. Attributes: <c>mode</c> ("max" | "avg"), <c>kernel</c> long[2],
    /// <c>strides</c> (default = 1,1), <c>pads</c> long[4] (top, left, bottom, right), <c>dilations</c> (max only),
    /// <c>ceil_mode</c> (bool), <c>count_include_pad</c> (bool, avg only). Padding never wins a max.
    /// Global pooling is <see cref="Reduce"/> over axes [2, 3].
    /// </summary>
    public static readonly OpSchema Pool = new("Pool", OpCategory.Reduction, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank != 4) throw new ArgumentException($"Pool expects x [N,C,H,W], got {x}.");
        var k = a.Get<long[]>("kernel");
        var (ho, wo) = SpatialOut(x[2].Fixed, x[3].Fixed, k[0], k[1], a, a.Get("ceil_mode", false));
        return [new(DataType.F32, new Shape(x[0], x[1], ho, wo))];
    });

    /// <summary>
    /// Reduction over arbitrary axes (never the symbolic outermost axis).
    /// Attributes: <c>mode</c> ("sum" | "mean" | "max" | "min" | "prod" | "l1" | "l2" | "sumsquare" | "logsumexp"),
    /// <c>axes</c> long[], <c>keepdims</c> (bool, default true).
    /// </summary>
    public static readonly OpSchema Reduce = new("Reduce", OpCategory.Reduction, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        var axes = a.Get<long[]>("axes").Select(ax => (int)(ax < 0 ? ax + x.Rank : ax)).ToHashSet();
        if (axes.Contains(0) && x[0].IsSymbolic) throw new ArgumentException("Reduce over the symbolic outermost axis is not supported.");
        bool keep = a.Get("keepdims", true);
        var dims = new List<Dim>();
        for (int d = 0; d < x.Rank; d++)
            if (!axes.Contains(d)) dims.Add(x[d]);
            else if (keep) dims.Add(1);
        return [new(DataType.F32, new Shape(dims))];
    });

    /// <summary>
    /// Matrix product of two activations with broadcast batch dims: a [..., M, K] · b [..., K, N] → [..., M, N].
    /// Attributes: <c>trans_a</c>, <c>trans_b</c> (bool; b is [..., N, K] when trans_b), <c>alpha</c> (float, default 1).
    /// For activation × weight use <see cref="MatMul"/>.
    /// </summary>
    public static readonly OpSchema BatchMatMul = new("BatchMatMul", OpCategory.MatMul, 2, 2, (i, a) =>
    {
        var x = i[0].Shape; var y = i[1].Shape;
        if (x.Rank < 2 || y.Rank < 2) throw new ArgumentException("BatchMatMul inputs must be rank >= 2.");
        bool ta = a.Get("trans_a", false), tb = a.Get("trans_b", false);
        Dim m = ta ? x[-1] : x[-2], k = ta ? x[-2] : x[-1];
        Dim k2 = tb ? y[-1] : y[-2], n = tb ? y[-2] : y[-1];
        if (k != k2) throw new ArgumentException($"BatchMatMul K mismatch: a{x} b{y} (trans_a={ta}, trans_b={tb}).");
        var batch = BroadcastShapes(new Shape(x.Dims.Take(x.Rank - 2)), new Shape(y.Dims.Take(y.Rank - 2)));
        return [new(DataType.F32, new Shape([.. batch.Dims, m, n]))];
    });

    /// <summary>
    /// Pads fixed (non-outermost-symbolic) axes. Attributes: <c>pads</c> long[2·rank] (begin per axis, then end
    /// per axis, ONNX order; negative crops), <c>mode</c> ("constant" | "reflect" | "edge"), <c>value</c> (float).
    /// </summary>
    public static readonly OpSchema Pad = new("Pad", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        var p = a.Get<long[]>("pads");
        if (p.Length != 2 * x.Rank) throw new ArgumentException($"Pad: pads must have {2 * x.Rank} entries.");
        var dims = new Dim[x.Rank];
        for (int d = 0; d < x.Rank; d++)
        {
            if (x[d].IsSymbolic) { if (p[d] != 0 || p[d + x.Rank] != 0) throw new ArgumentException("Pad on a symbolic axis."); dims[d] = x[d]; }
            else dims[d] = x[d].Fixed + p[d] + p[d + x.Rank];
        }
        return [new(DataType.F32, new Shape(dims))];
    });

    /// <summary>
    /// Spatial resize of x [N, C, H, W]. Attributes: <c>size</c> long[2] (output H, W), <c>mode</c>
    /// ("nearest" | "linear"), <c>coord</c> ("half_pixel" | "align_corners" | "asymmetric" | "pytorch_half_pixel"),
    /// <c>nearest_mode</c> ("round_prefer_floor" | "floor" | ...).
    /// </summary>
    public static readonly OpSchema Resize = new("Resize", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        var s = a.Get<long[]>("size");
        return [new(DataType.F32, new Shape(x[0], x[1], s[0], s[1]))];
    });

    static (long H, long W) SpatialOut(long h, long w, long kh, long kw, Attributes a, bool ceil)
    {
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        long Out(long n, long k, long s, long p0, long p1, long d)
        {
            long span = n + p0 + p1 - d * (k - 1) - 1;
            long o = (ceil ? (span + s - 1) / s : span / s) + 1;
            // ONNX/PyTorch: with ceil_mode the last window must start inside the (left-padded) input.
            if (ceil && (o - 1) * s >= n + p0) o--;
            return o;
        }
        return (Out(h, kh, st[0], pd[0], pd[2], dl[0]), Out(w, kw, st[1], pd[1], pd[3], dl[1]));
    }
}
