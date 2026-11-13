using Sharplight.Tensors;

namespace Sharplight.Graph;

// Classical machine learning (decision forests, kernel machines).
public static partial class StandardOps
{
    /// <summary>
    /// Decision forest: x f32 <c>[N, F]</c> → scores f32 <c>[N, n_targets]</c> and hits f32 <c>[N, n_targets]</c> (1 where any leaf contributed
    /// to the target, else 0; the score is then 0 too). Base values and post-transforms are left to the consumer. Nodes of all trees share one table
    /// indexed by node: <c>node_mode</c> (long[]: 0 x ≤ v, 1 x &lt; v, 2 x ≥ v, 3 x &gt; v, 4 x = v, 5 x ≠ v, 6 x ∈ set, 7 leaf),
    /// <c>node_feature</c> (long[]), <c>node_value</c> (double[]: threshold), <c>node_true</c> / <c>node_false</c> (long[]: child node),
    /// <c>node_missing_true</c> (long[]: a NaN feature takes the true branch); sets are <c>member_values</c> (double[]) between
    /// <c>member_offsets</c> (long[], nodes + 1). A leaf adds <c>leaf_weights</c> (float[]) to <c>leaf_targets</c> (long[]) between
    /// <c>leaf_offsets</c> (long[], nodes + 1). <c>roots</c> (long[]): root node of each tree. <c>aggregate</c> ("SUM", "AVERAGE": sum / trees,
    /// "MIN", "MAX": over the contributions to a target).
    /// </summary>
    public static readonly OpSchema TreeEnsemble = new("TreeEnsemble", OpCategory.Composite, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank != 2)
        {
            throw new ArgumentException($"TreeEnsemble expects x [N, F], got {x}.");
        }

        var y = new Shape(x[0], a.Get<long>("n_targets"));
        return [new(DataType.F32, y), new(DataType.F32, y)];
    });

    /// <summary>
    /// Elementwise dictionary lookup: y = values[k] where x = keys[k], else <c>default</c>. x f32 or i32 (any shape) → <c>type</c> (f32 or i32),
    /// same shape. Attributes: <c>keys</c>, <c>values</c> (double[], same length; a NaN key matches NaN), <c>default</c> (double),
    /// <c>type</c> (DataType name).
    /// </summary>
    public static readonly OpSchema LabelEncode = new("LabelEncode", OpCategory.Data, 1, 1, (i, a) =>
        [new(DataType.Get(a.Get("type", "f32")), i[0].Shape)]);

    /// <summary>
    /// One-vs-one combination of pairwise decisions (libsvm order: pairs (0,1), (0,2), ..., (1,2), ...): d f32 <c>[N, k(k-1)/2]</c> →
    /// winner i32 <c>[N]</c> (most votes, a pair votes for its first class when d &gt; 0; ties go to the lower class) and, when
    /// <c>prob_a</c> / <c>prob_b</c> (float[] per pair) are given, class probabilities f32 <c>[N, k]</c> by Platt scaling of each pair and
    /// libsvm's pairwise coupling. Attribute <c>classes</c> (long: k).
    /// </summary>
    public static readonly OpSchema OneVsOne = new("OneVsOne", OpCategory.Composite, 1, 1, (i, a) =>
    {
        var d = i[0].Shape;
        long k = a.Get<long>("classes");
        if (d.Rank != 2 || d[1].Fixed != k * (k - 1) / 2)
        {
            throw new ArgumentException($"OneVsOne expects decisions [N, {k * (k - 1) / 2}] for {k} classes, got {d}.");
        }

        var winner = new TensorType(DataType.I32, new Shape(d[0]));
        return a.Has("prob_a") ? [winner, new(DataType.F32, new Shape(d[0], k))] : [winner];
    });
}
